# batch_size = encoder_hidden_states.size(0)
        # target_length = position_ids.size(1) if position_ids is not None else bar_ids.size(1)

        # # Initialize decoder input with BOS token instead of zeros
        # bos_token = torch.full((batch_size, 1), self.vocab.to_i(BOS_TOKEN), device=encoder_hidden_states.device)
        # decoder_input = self.in_layer(bos_token)  # [batch_size, 1, d_model]

        # # Get global conditioning from encoder (average pooling over sequence length)
        # global_condition = encoder_hidden_states.mean(dim=1, keepdim=True)  # [batch_size, 1, d_model]

        # # Generate different feature types
        # bar_symbolic_logits = self.bar_symbolic_out(hidden)  # [batch_size, target_seq_len, vocab_size]
        # latent_features = self.latent_out(hidden)  # [batch_size, target_seq_len, d_latent]

        # # Generate sequence step by step
        # outputs = []
        # for i in range(target_length):
        #     # Get current position embeddings
        #     if bar_ids is not None:
        #         curr_bar_emb = self.bar_embedding(bar_ids[:, i : i + 1])
        #         decoder_input = decoder_input + curr_bar_emb
        #     if position_ids is not None:
        #         curr_pos_emb = self.pos_embedding(position_ids[:, i : i + 1])
        #         decoder_input = decoder_input + curr_pos_emb

        #     # # Add explicit conditioning by concatenating global condition
        #     # decoder_input = decoder_input + global_condition
        #     decoder_input = torch.cat([decoder_input, encoder_hidden_states], dim=-1)

        #     # Run decoder for current step
        #     decoder_out = self.transformer.decoder(inputs_embeds=decoder_input, encoder_hidden_states=encoder_hidden_states, output_hidden_states=True)
        #     hidden = decoder_out.hidden_states[-1]
        #     outputs.append(hidden)

        #     # Update decoder input for next step
        #     decoder_input = hidden

        # # Concatenate all outputs
        # hidden = torch.cat(outputs, dim=1)  # [batch_size, target_seq_len, d_model]

        # # Generate different feature types
        # bar_symbolic_logits = self.bar_symbolic_out(hidden)  # [batch_size, target_seq_len, vocab_size]
        # latent_features = self.latent_out(hidden)  # [batch_size, target_seq_len, d_latent]

        # return {
        #     "bar_symbolic_features": bar_symbolic_logits,
        #     "latents": latent_features,
        # }
