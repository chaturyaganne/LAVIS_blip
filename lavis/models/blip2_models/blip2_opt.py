"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Modified BLIP2 OPT model with VisAlign architecture
 File location: lavis/models/blip2_models/blip2_opt_visalign.py
"""
import logging
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig
import transformers


@registry.register_model("blip2_opt_visalign")
class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model with VisAlign modifications.
    
    VisAlign Architecture:
    1. Visual Encoder → Q-Former → Visual Embeddings (V)
    2. Average Visual Embeddings: V_avg = mean(V, dim=1)
    3. Text Tokenizer → Text Embeddings (T)
    4. Concatenate: [T, V_avg] along feature dimension
    5. Linear Transformation: T_new = Linear([T, V_avg])
    6. Input to LLM: [V, T_new]

    Supported model types:
        - pretrain_opt2.7b: pretrained model with OPT2.7b
        - pretrain_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: finetuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: finetuned image captioning model with OPT6.7b
        
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt_visalign", "pretrain_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b_visalign.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b_visalign.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b_visalign.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b_visalign.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        Args:
            vit_model (str): Vision transformer model name
            img_size (int): Input image size
            drop_path_rate (float): Drop path rate for ViT
            use_grad_checkpoint (bool): Use gradient checkpointing for ViT
            vit_precision (str): ViT precision ("fp16" or "fp32")
            freeze_vit (bool): Freeze vision encoder
            num_query_token (int): Number of query tokens for Q-Former
            opt_model (str): OPT model name/path
            prompt (str): Prompt template for generation
            max_txt_len (int): Maximum text length
            apply_lemmatizer (bool): Apply lemmatization to outputs
        """
        super().__init__()

        # Initialize vision encoder
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # Initialize Q-Former
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # Initialize OPT model
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        
        # Freeze OPT model
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        # Original projection layer from Q-Former to OPT embedding space
        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        # ===== VisAlign Component =====
        # Linear layer to fuse text embeddings with averaged visual embeddings
        # Input: [text_embed_dim + visual_embed_dim] = [hidden_size + hidden_size]
        # Output: [text_embed_dim] = [hidden_size]
        self.text_visual_fusion = nn.Linear(
            self.opt_model.config.hidden_size * 2,  # Concatenated dimension
            self.opt_model.config.hidden_size       # Output dimension
        )
        
        # Initialize fusion layer weights
        nn.init.xavier_uniform_(self.text_visual_fusion.weight)
        nn.init.zeros_(self.text_visual_fusion.bias)
        
        logging.info(f"Initialized VisAlign fusion layer: {self.opt_model.config.hidden_size * 2} -> {self.opt_model.config.hidden_size}")

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

    def forward(self, samples):
        """
        Forward pass with VisAlign architecture
        
        Args:
            samples (dict): Dictionary containing:
                - image (torch.Tensor): Batch of images [B, C, H, W]
                - text_input (list): List of text strings, length B
                
        Returns:
            dict: Dictionary containing 'loss'
        """
        image = samples["image"]
        text = [t + "\n" for t in samples["text_input"]]

        # ===== Step 1: Process Visual Embeddings =====
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # Q-Former processes image embeddings
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # Project Q-Former output to OPT embedding space
        inputs_opt = self.opt_proj(query_output.last_hidden_state)  # [B, num_query, D]
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        # ===== Step 2: Average Visual Embeddings (VisAlign) =====
        visual_avg = inputs_opt.mean(dim=1, keepdim=True)  # [B, 1, D]

        # ===== Step 3: Process Text Input =====
        self.opt_tokenizer.padding_side = "right"
        text_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        # Get text embeddings from OPT embedding layer
        text_embeds = self.opt_model.model.decoder.embed_tokens(
            text_tokens.input_ids
        )  # [B, seq_len, D]

        # ===== Step 4: Concatenate Visual Average to Text Embeddings (VisAlign) =====
        B, T, D = text_embeds.shape
        # Expand visual_avg to match text sequence length
        visual_avg_expanded = visual_avg.expand(B, T, D)  # [B, seq_len, D]
        
        # Concatenate along feature dimension
        text_visual_concat = torch.cat(
            [text_embeds, visual_avg_expanded], dim=-1
        )  # [B, seq_len, 2*D]

        # ===== Step 5: Apply Linear Transformation (VisAlign) =====
        text_embeds_transformed = self.text_visual_fusion(
            text_visual_concat
        )  # [B, seq_len, D]

        # ===== Step 6: Prepare Final Input to LLM =====
        # Concatenate original visual embeddings with transformed text embeddings
        inputs_embeds = torch.cat(
            [inputs_opt, text_embeds_transformed], dim=1
        )  # [B, num_query + seq_len, D]
        
        attention_mask = torch.cat(
            [atts_opt, text_tokens.attention_mask], dim=1
        )

        # Prepare targets for language modeling
        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        # ===== Step 7: Forward through LLM =====
        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Generate text with VisAlign architecture
        
        Args:
            samples (dict): Dictionary containing:
                - image (torch.Tensor): Batch of images [B, C, H, W]
                - prompt (str, optional): Text prompt for generation
            use_nucleus_sampling (bool): Use nucleus sampling
            num_beams (int): Number of beams for beam search
            max_length (int): Maximum length of generated text
            min_length (int): Minimum length of generated text
            top_p (float): Top-p sampling parameter
            repetition_penalty (float): Repetition penalty
            length_penalty (float): Length penalty
            num_captions (int): Number of captions to generate per image
            temperature (float): Temperature for sampling
            
        Returns:
            list: List of generated text strings
        """
        image = samples["image"]
        
        # ===== Step 1: Process Visual Embeddings =====
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # Project to OPT space
        inputs_opt = self.opt_proj(query_output.last_hidden_state)  # [B, num_query, D]
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        # ===== Step 2: Average Visual Embeddings (VisAlign) =====
        visual_avg = inputs_opt.mean(dim=1, keepdim=True)  # [B, 1, D]

        # ===== Step 3: Process Prompt =====
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        prompt = [prompt] * image.size(0)

        opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt").to(image.device)
        input_ids = opt_tokens.input_ids
        
        # Get prompt embeddings
        prompt_embeds = self.opt_model.model.decoder.embed_tokens(input_ids)  # [B, prompt_len, D]

        # ===== Step 4: Apply VisAlign Transformation to Prompt =====
        B, T, D = prompt_embeds.shape
        visual_avg_expanded = visual_avg.expand(B, T, D)
        prompt_visual_concat = torch.cat([prompt_embeds, visual_avg_expanded], dim=-1)
        prompt_embeds_transformed = self.text_visual_fusion(prompt_visual_concat)

        # ===== Step 5: Prepare Final Input =====
        inputs_embeds = torch.cat([inputs_opt, prompt_embeds_transformed], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        # ===== Step 6: Generate =====
        outputs = self.opt_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        
        output_text = self.opt_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        output_text = [text.strip() for text in output_text]
        
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):
        """
        Predict answers for VQA task
        
        Args:
            samples (dict): Dictionary containing:
                - image (torch.Tensor): Batch of images
                - text_input (list or str): Questions
            num_beams (int): Number of beams
            inference_method (str): "generate" or "rank"
            max_len (int): Maximum answer length
            min_len (int): Minimum answer length
            num_ans_candidates (int): Number of answer candidates for ranking
            answer_list (list): List of candidate answers for ranking
            prompt (str): Prompt template
            length_penalty (float): Length penalty
            
        Returns:
            list: List of predicted answers
        """
        image = samples["image"]
        text_input = samples["text_input"]

        # Process image
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)
        
        # Average visual embeddings
        visual_avg = inputs_opt.mean(dim=1, keepdim=True)

        # Process question
        if isinstance(text_input, str):
            text_input = [text_input] * image.size(0)
        if prompt:
            text_input = [prompt.format(question) for question in text_input]

        self.opt_tokenizer.padding_side = "right"
        opt_tokens = self.opt_tokenizer(
            text_input,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        # Apply VisAlign to question embeddings
        question_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        B, T, D = question_embeds.shape
        visual_avg_expanded = visual_avg.expand(B, T, D)
        question_visual_concat = torch.cat([question_embeds, visual_avg_expanded], dim=-1)
        question_embeds_transformed = self.text_visual_fusion(question_visual_concat)

        inputs_embeds = torch.cat([inputs_opt, question_embeds_transformed], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        if inference_method == "generate":
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        else:
            raise NotImplementedError("Ranking inference not implemented for VisAlign")

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        """Apply lemmatization to answers"""
        def apply(answer):
            doc = self.lemmatizer(answer)
            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)
            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        """Lazy initialization of lemmatizer"""
        if self._lemmatizer is None:
            try:
                import spacy
                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    """
                )
        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        """Create model from config"""
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )

        model.load_checkpoint_from_config(cfg)

        return model
