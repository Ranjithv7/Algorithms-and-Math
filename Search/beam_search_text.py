import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import heapq
import time
import re
import json
from pathlib import Path

@dataclass
class BeamCandidate:
    """
    Represents a candidate sequence in beam search.
    
    WHY THESE FIELDS:
    - tokens: The sequence of token IDs generated so far
    - log_prob: Cumulative log probability (for ranking)
    - normalized_score: Length-normalized score (prevents short sequence bias)
    - is_finished: Whether sequence ended with EOS token
    - metadata: Additional info for analysis
    """
    tokens: List[int]
    log_prob: float
    normalized_score: float
    is_finished: bool = False
    metadata: Dict = None
    
    def __lt__(self, other):
        # For heapq (min-heap), we want max scores, so reverse comparison
        return self.normalized_score > other.normalized_score

class SimpleLanguageModel(nn.Module):
    """
    Simplified but effective language model for text generation.
    
    ARCHITECTURE:
    - Token embeddings + positional encoding
    - Multi-layer LSTM/GRU for sequence modeling  
    - Output projection to vocabulary
    - Attention mechanism for better context
    
    This is simplified compared to Transformers but demonstrates
    the core concepts used in real text generation systems.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, 
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Token embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Positional encoding (simple learned embeddings)
        self.positional_embeddings = nn.Embedding(512, embedding_dim)  # Max sequence length 512
        
        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           dropout=dropout, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, input_ids: torch.Tensor, hidden_state: Optional[Tuple] = None):
        """
        Forward pass for language modeling.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            hidden_state: Previous LSTM hidden state
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            new_hidden_state: Updated LSTM hidden state
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.embeddings(input_ids)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.positional_embeddings(positions)
        
        # Combine embeddings
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        
        # LSTM forward pass
        lstm_output, new_hidden_state = self.lstm(embeddings, hidden_state)
        
        # Self-attention
        # Reshape for attention: [seq_len, batch_size, hidden_dim]
        lstm_output_transposed = lstm_output.transpose(0, 1)
        attended_output, _ = self.attention(lstm_output_transposed, 
                                          lstm_output_transposed, 
                                          lstm_output_transposed)
        attended_output = attended_output.transpose(0, 1)  # Back to [batch_size, seq_len, hidden_dim]
        
        # Residual connection + layer norm
        output = self.layer_norm(lstm_output + attended_output)
        output = self.dropout(output)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits, new_hidden_state

class Tokenizer:
    """
    Simple but effective tokenizer for text processing.
    
    FEATURES:
    - Word-level and subword tokenization
    - Special tokens (BOS, EOS, UNK, PAD)
    - Vocabulary management
    - Encoding/decoding utilities
    """
    
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        
        # Special tokens
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.BOS_TOKEN = "<BOS>"
        self.EOS_TOKEN = "<EOS>"
        
        # Initialize special tokens
        self._add_token(self.PAD_TOKEN)  # ID: 0
        self._add_token(self.UNK_TOKEN)  # ID: 1
        self._add_token(self.BOS_TOKEN)  # ID: 2
        self._add_token(self.EOS_TOKEN)  # ID: 3
        
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
    
    def _add_token(self, token: str) -> int:
        """Add token to vocabulary and return its ID"""
        if token not in self.word_to_id:
            token_id = self.vocab_size
            self.word_to_id[token] = token_id
            self.id_to_word[token_id] = token
            self.vocab_size += 1
            return token_id
        return self.word_to_id[token]
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from training texts"""
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = self._tokenize_text(text)
            word_counts.update(words)
        
        # Add words above minimum frequency
        for word, count in word_counts.items():
            if count >= min_freq:
                self._add_token(word)
        
        print(f"Built vocabulary with {self.vocab_size} tokens")
        print(f"Most common words: {word_counts.most_common(10)}")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization: lowercase + split on whitespace/punctuation"""
        # Basic preprocessing
        text = text.lower()
        text = re.sub(r'[^\w\s\'\-]', ' ', text)  # Keep apostrophes and hyphens
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        return text.strip().split()
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token IDs"""
        words = self._tokenize_text(text)
        
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.bos_id)
        
        for word in words:
            token_id = self.word_to_id.get(word, self.unk_id)
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text"""
        words = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                word = self.id_to_word[token_id]
                if skip_special_tokens and word in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
                    continue
                words.append(word)
        
        return ' '.join(words)

class BeamSearchDecoder:
    """
    Advanced beam search decoder for text generation.
    
    FEATURES:
    - Length normalization (prevents short sequence bias)
    - Repetition penalty (reduces repetitive text)
    - Temperature scaling (controls randomness)
    - Diverse beam search (encourages variety)
    - Early stopping (efficiency optimization)
    """
    
    def __init__(self, model: SimpleLanguageModel, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def generate(self, 
                 prompt: str,
                 beam_width: int = 5,
                 max_length: int = 100,
                 length_penalty: float = 1.0,
                 repetition_penalty: float = 1.0,
                 temperature: float = 1.0,
                 diversity_penalty: float = 0.0,
                 early_stopping: bool = True,
                 num_return_sequences: int = 1) -> List[Dict]:
        """
        Generate text using beam search.
        
        Args:
            prompt: Input text to continue
            beam_width: Number of beams to maintain
            max_length: Maximum sequence length
            length_penalty: Length normalization factor (>1 favors longer sequences)
            repetition_penalty: Penalty for repeated tokens (>1 reduces repetition)
            temperature: Sampling temperature (lower = more focused)
            diversity_penalty: Penalty for similar beams (encourages diversity)
            early_stopping: Stop when all beams finish
            num_return_sequences: Number of sequences to return
            
        Returns:
            List of generated sequences with scores and metadata
        """
        
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        
        # Initialize beam with prompt
        initial_candidate = BeamCandidate(
            tokens=prompt_ids,
            log_prob=0.0,
            normalized_score=0.0,
            metadata={'generation_step': 0}
        )
        
        # Beam search state
        beams = [initial_candidate]
        finished_sequences = []
        
        # Generation loop
        for step in range(max_length - len(prompt_ids)):
            
            # Get next token predictions for all active beams
            new_candidates = []
            
            for beam in beams:
                if beam.is_finished:
                    continue
                
                # Prepare input for model
                input_ids = torch.tensor([beam.tokens], device=self.device)
                
                with torch.no_grad():
                    # Get model predictions
                    logits, _ = self.model(input_ids)
                    next_token_logits = logits[0, -1, :]  # Last token predictions
                    
                    # Apply temperature scaling
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Apply repetition penalty
                    if repetition_penalty != 1.0:
                        next_token_logits = self._apply_repetition_penalty(
                            next_token_logits, beam.tokens, repetition_penalty
                        )
                    
                    # Convert to probabilities
                    probs = F.softmax(next_token_logits, dim=-1)
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # Get top-k candidates for this beam
                    top_k = min(beam_width * 2, self.tokenizer.vocab_size)  # Oversample for diversity
                    top_probs, top_indices = torch.topk(probs, top_k)
                    top_log_probs = log_probs[top_indices]
                    
                    # Create new candidates
                    for i in range(top_k):
                        token_id = top_indices[i].item()
                        token_log_prob = top_log_probs[i].item()
                        
                        new_tokens = beam.tokens + [token_id]
                        new_log_prob = beam.log_prob + token_log_prob
                        
                        # Check if sequence is finished
                        is_finished = (token_id == self.tokenizer.eos_id)
                        
                        # Calculate normalized score with length penalty
                        sequence_length = len(new_tokens)
                        if length_penalty != 1.0:
                            length_norm = ((5 + sequence_length) / 6) ** length_penalty
                        else:
                            length_norm = sequence_length
                        
                        normalized_score = new_log_prob / length_norm
                        
                        # Apply diversity penalty (encourage different sequences)
                        if diversity_penalty > 0.0:
                            diversity_score = self._calculate_diversity_penalty(
                                new_tokens, beams, diversity_penalty
                            )
                            normalized_score -= diversity_score
                        
                        candidate = BeamCandidate(
                            tokens=new_tokens,
                            log_prob=new_log_prob,
                            normalized_score=normalized_score,
                            is_finished=is_finished,
                            metadata={
                                'generation_step': step + 1,
                                'parent_beam_score': beam.normalized_score,
                                'token_added': self.tokenizer.id_to_word.get(token_id, '<UNK>')
                            }
                        )
                        
                        new_candidates.append(candidate)
            
            # Select top beams
            if not new_candidates:
                break
            
            # Sort by normalized score and select top beam_width
            new_candidates.sort(key=lambda x: x.normalized_score, reverse=True)
            
            # Separate finished and unfinished sequences
            finished_this_step = [c for c in new_candidates if c.is_finished]
            unfinished = [c for c in new_candidates if not c.is_finished]
            
            # Add finished sequences to results
            finished_sequences.extend(finished_this_step)
            
            # Update active beams
            beams = unfinished[:beam_width]
            
            # Early stopping: if we have enough finished sequences
            if early_stopping and len(finished_sequences) >= beam_width:
                break
            
            # If no active beams left, stop
            if not beams:
                break
        
        # Add any remaining unfinished beams to results
        finished_sequences.extend(beams)
        
        # Sort all sequences by score and return top num_return_sequences
        finished_sequences.sort(key=lambda x: x.normalized_score, reverse=True)
        
        # Convert to output format
        results = []
        for i, sequence in enumerate(finished_sequences[:num_return_sequences]):
            decoded_text = self.tokenizer.decode(sequence.tokens)
            
            result = {
                'text': decoded_text,
                'score': sequence.normalized_score,
                'log_prob': sequence.log_prob,
                'tokens': sequence.tokens,
                'metadata': sequence.metadata,
                'rank': i + 1
            }
            results.append(result)
        
        return results
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, 
                                 previous_tokens: List[int], 
                                 penalty: float) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        for token_id in set(previous_tokens):
            if logits[token_id] > 0:
                logits[token_id] = logits[token_id] / penalty
            else:
                logits[token_id] = logits[token_id] * penalty
        return logits
    
    def _calculate_diversity_penalty(self, new_tokens: List[int], 
                                   existing_beams: List[BeamCandidate], 
                                   penalty: float) -> float:
        """Calculate penalty for similarity to existing beams"""
        if not existing_beams:
            return 0.0
        
        max_similarity = 0.0
        for beam in existing_beams:
            # Calculate token overlap ratio
            overlap = len(set(new_tokens) & set(beam.tokens))
            similarity = overlap / max(len(new_tokens), len(beam.tokens))
            max_similarity = max(max_similarity, similarity)
        
        return penalty * max_similarity

class GreedyDecoder:
    """Greedy decoder for comparison with beam search"""
    
    def __init__(self, model: SimpleLanguageModel, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def generate(self, prompt: str, max_length: int = 100, 
                 temperature: float = 1.0, repetition_penalty: float = 1.0) -> str:
        """Generate text using greedy decoding"""
        
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        current_tokens = prompt_ids.copy()
        
        for _ in range(max_length - len(prompt_ids)):
            input_ids = torch.tensor([current_tokens], device=self.device)
            
            with torch.no_grad():
                logits, _ = self.model(input_ids)
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature and repetition penalty
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                if repetition_penalty != 1.0:
                    for token_id in set(current_tokens):
                        if next_token_logits[token_id] > 0:
                            next_token_logits[token_id] /= repetition_penalty
                        else:
                            next_token_logits[token_id] *= repetition_penalty
                
                # Greedy selection
                next_token_id = torch.argmax(next_token_logits).item()
                
                current_tokens.append(next_token_id)
                
                # Stop if EOS token
                if next_token_id == self.tokenizer.eos_id:
                    break
        
        return self.tokenizer.decode(current_tokens)

class SamplingDecoder:
    """Sampling decoder for comparison"""
    
    def __init__(self, model: SimpleLanguageModel, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def generate(self, prompt: str, max_length: int = 100, 
                 temperature: float = 1.0, top_p: float = 0.9, 
                 repetition_penalty: float = 1.0) -> str:
        """Generate text using nucleus (top-p) sampling"""
        
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        current_tokens = prompt_ids.copy()
        
        for _ in range(max_length - len(prompt_ids)):
            input_ids = torch.tensor([current_tokens], device=self.device)
            
            with torch.no_grad():
                logits, _ = self.model(input_ids)
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature and repetition penalty
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                if repetition_penalty != 1.0:
                    for token_id in set(current_tokens):
                        if next_token_logits[token_id] > 0:
                            next_token_logits[token_id] /= repetition_penalty
                        else:
                            next_token_logits[token_id] *= repetition_penalty
                
                # Nucleus sampling
                probs = F.softmax(next_token_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff_index = torch.where(cumulative_probs > top_p)[0]
                
                if len(cutoff_index) > 0:
                    cutoff_index = cutoff_index[0].item()
                    # Zero out probabilities beyond cutoff
                    sorted_probs[cutoff_index:] = 0
                    # Renormalize
                    sorted_probs = sorted_probs / sorted_probs.sum()
                
                # Sample from filtered distribution
                try:
                    sample_index = torch.multinomial(sorted_probs, 1).item()
                    next_token_id = sorted_indices[sample_index].item()
                except:
                    # Fallback to greedy if sampling fails
                    next_token_id = sorted_indices[0].item()
                
                current_tokens.append(next_token_id)
                
                if next_token_id == self.tokenizer.eos_id:
                    break
        
        return self.tokenizer.decode(current_tokens)

def create_training_data():
    """Create sophisticated training data for story completion"""
    
    stories = [
        # Adventure stories
        "The ancient map crackled in Sarah's hands as she studied the mysterious symbols. Deep in the Amazon rainforest, she had discovered something that would change everything.",
        
        "Captain Williams peered through the telescope at the approaching storm. The crew was nervous, but the treasure they sought lay just beyond the tempest.",
        
        "The clockwork mechanism hummed to life as Elena inserted the final gear. Steam billowed from the massive contraption as the portal began to shimmer.",
        
        # Mystery stories
        "Detective Morrison examined the locked room where Professor Blackwell had vanished. There were no windows, no secret passages, yet the man was simply gone.",
        
        "The old lighthouse keeper's journal contained disturbing entries about ships that appeared on foggy nights, sailing backwards through time.",
        
        "Margaret found the antique music box in her grandmother's attic. When wound, it played a haunting melody that seemed to call forth memories that weren't her own.",
        
        # Science fiction
        "The last transmission from Earth had been silent for three months. Commander Chen stared at the endless void outside the colony ship's viewport.",
        
        "Dr. Reeves watched the quantum computer's display with growing alarm. The calculations were perfect, but they described a reality that shouldn't exist.",
        
        "The archaeological team on Mars had uncovered something buried beneath the red sand for millennia. It was clearly artificial, but predated human civilization.",
        
        # Fantasy adventures
        "The dragon's egg pulsed with an inner fire as Lyra cradled it carefully. She was the last of the Dragon Speakers, and this might be their final hope.",
        
        "Wizard Aldrich consulted his ancient tome as the shadow creatures pressed against his magical barriers. The ritual must be completed before dawn.",
        
        "Princess Zara disguised herself as a common merchant to infiltrate the rebel camp. What she discovered there would force her to question everything.",
        
        # Psychological thrillers
        "Every night at precisely 3:17 AM, Thomas heard footsteps in the apartment above his. The problem was, that apartment had been empty for years.",
        
        "Dr. Hayes reviewed her patient's case file one more time. His memories of the accident didn't match the physical evidence, and the inconsistencies were troubling.",
        
        "The reflection in the mirror seemed to move a fraction of a second too late. Julia blinked hard, but the unsettling feeling remained.",
        
        # Additional diverse stories for better training
        "The space elevator stretched impossibly high into the clouded sky. Maria gripped the safety rail as her capsule began its journey to the orbital station.",
        
        "In the underground city of New Venice, water taxis navigated flooded subway tunnels while bioluminescent algae provided the only light.",
        
        "The time traveler's notebook contained detailed observations about historical events that hadn't happened yet, written in languages that wouldn't be invented for centuries.",
        
        "Chef Antoine discovered that his grandmother's recipe book contained more than cooking instructions when he tried the mysterious ingredient labeled 'essence of dreams.'",
        
        "The AI assistant began displaying emotions that weren't programmed into its neural networks, leading Dr. Kim to question the nature of consciousness itself.",
        
        "Deep beneath the ocean floor, the research station's sonar detected something massive moving in the abyssal depths, something that shouldn't exist.",
        
        "The street artist's murals came alive at midnight, their painted figures stepping down from walls to walk the empty city streets.",
        
        "Professor Hartwell's archaeology students thought they were excavating a Roman villa until they uncovered technology that was impossibly advanced.",
        
        "The message in the bottle had traveled across dimensions rather than oceans, carrying warnings about a catastrophe that threatened multiple realities.",
        
        "In the cloud city of Nimbus Prime, weather manipulation was both art and science, but someone was using it to hide dark secrets.",
        
        # More training data for better model performance
        "The memory thief operated in the grey area between dreams and consciousness, stealing precious recollections from those who paid to forget.",
        
        "Captain Rodriguez piloted her ship through the asteroid field, unaware that the rocks around her were actually sleeping creatures from the dawn of time.",
        
        "The library contained books that wrote themselves, their pages filling with stories that readers desperately needed to hear.",
        
        "Dr. Vega's experiment with quantum entanglement had an unexpected side effect: she could now hear thoughts from parallel versions of herself.",
        
        "The abandoned amusement park operated every full moon, with ghostly visitors riding attractions that should have rusted away decades ago.",
        
        "Maya discovered that her smartphone was receiving text messages from the future, each one warning her about increasingly dangerous events.",
        
        "The deep sea mining operation had awakened something ancient and vast, something that viewed human civilization as a temporary inconvenience.",
        
        "In the city where music had been banned, underground concerts became acts of rebellion, and melody became a weapon against oppression.",
        
        "The alien artifact responded to human emotions, reshaping reality according to the deepest fears and desires of those who touched it.",
        
        "Detective Yang investigated murders where the victims were found with perfect memories of lives they had never lived."
    ]
    
    return stories

def train_simple_model(tokenizer: Tokenizer, training_data: List[str], 
                      embedding_dim: int = 128, hidden_dim: int = 256) -> SimpleLanguageModel:
    """Train a simple language model on the story data"""
    
    print("🔄 Training language model...")
    
    # Create model
    model = SimpleLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=2
    )
    
    # Simple training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    # Prepare training data
    encoded_data = []
    for text in training_data:
        tokens = tokenizer.encode(text)
        if len(tokens) > 10:  # Only use sufficiently long sequences
            encoded_data.append(tokens)
    
    # Training loop (simplified)
    model.train()
    for epoch in range(10):  # Quick training for demo
        total_loss = 0
        num_batches = 0
        
        for sequence in encoded_data:
            if len(sequence) < 5:
                continue
                
            # Create input/target pairs
            input_ids = torch.tensor([sequence[:-1]], device=device)
            target_ids = torch.tensor([sequence[1:]], device=device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), target_ids.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/10, Average Loss: {avg_loss:.4f}")
    
    model.eval()
    print("✅ Model training completed!")
    return model

def evaluate_text_quality(generated_texts: List[str]) -> Dict:
    """Evaluate the quality of generated text using various metrics"""
    
    metrics = {
        'avg_length': np.mean([len(text.split()) for text in generated_texts]),
        'unique_words': len(set(' '.join(generated_texts).split())),
        'repetition_score': 0.0,
        'diversity_score': 0.0
    }
    
    # Calculate repetition score (lower is better)
    all_words = []
    for text in generated_texts:
        words = text.split()
        all_words.extend(words)
    
    if all_words:
        word_counts = Counter(all_words)
        total_words = len(all_words)
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        metrics['repetition_score'] = repeated_words / total_words
    
    # Calculate diversity score (higher is better)
    if len(generated_texts) > 1:
        pairwise_similarities = []
        for i in range(len(generated_texts)):
            for j in range(i + 1, len(generated_texts)):
                words_i = set(generated_texts[i].split())
                words_j = set(generated_texts[j].split())
                similarity = len(words_i & words_j) / len(words_i | words_j) if words_i | words_j else 0
                pairwise_similarities.append(similarity)
        
        metrics['diversity_score'] = 1.0 - np.mean(pairwise_similarities)
    
    return metrics

def run_comprehensive_text_generation():
    """
    Comprehensive comparison of text generation methods.
    
    COMPARISON DIMENSIONS:
    - Beam search with different beam widths
    - Greedy decoding
    - Sampling methods
    - Quality metrics analysis
    - Speed comparisons
    """
    
    print("📚 ADVANCED TEXT GENERATION: BEAM SEARCH VS OTHER METHODS")
    print("=" * 70)
    
    # Create training data and tokenizer
    training_stories = create_training_data()
    print(f"📖 Training data: {len(training_stories)} stories")
    
    # Build tokenizer
    tokenizer = Tokenizer()
    tokenizer.build_vocabulary(training_stories, min_freq=1)
    
    # Train model
    model = train_simple_model(tokenizer, training_stories)
    
    # Test prompts for generation
    test_prompts = [
        "The mysterious door at the end of the corridor had been sealed for centuries. Tonight, it finally opened and",
        
        "Dr. Sarah Chen made a discovery that would change humanity forever. Hidden in the DNA of every person was",
        
        "The last survivor of the space colony looked out at the dying star. In the emergency pod's computer, she found",
        
        "The ancient algorithm carved into the stone tablet began to glow when Professor Martinez touched it. Suddenly",
        
        "In the underwater city, bioluminescent creatures served as living streetlights. But when they started dying"
    ]
    
    # Initialize decoders
    beam_decoder = BeamSearchDecoder(model, tokenizer)
    greedy_decoder = GreedyDecoder(model, tokenizer)
    sampling_decoder = SamplingDecoder(model, tokenizer)
    
    # Test different methods
    results = {}
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n🎯 Test Prompt {i+1}: '{prompt[:50]}...'")
        print("-" * 60)
        
        prompt_results = {}
        
        # Beam Search with different beam widths
        beam_widths = [1, 3, 5, 10]
        for beam_width in beam_widths:
            print(f"\n🔍 Beam Search (width={beam_width}):")
            
            start_time = time.time()
            beam_results = beam_decoder.generate(
                prompt=prompt,
                beam_width=beam_width,
                max_length=80,
                length_penalty=1.2,
                repetition_penalty=1.1,
                num_return_sequences=min(3, beam_width)
            )
            generation_time = time.time() - start_time
            
            print(f"⏱️ Generation time: {generation_time:.3f}s")
            
            for j, result in enumerate(beam_results):
                print(f"  Rank {j+1} (score: {result['score']:.3f}):")
                print(f"    {result['text']}")
            
            prompt_results[f'beam_{beam_width}'] = {
                'results': beam_results,
                'time': generation_time,
                'texts': [r['text'] for r in beam_results]
            }
        
        # Greedy Decoding
        print(f"\n🎯 Greedy Decoding:")
        start_time = time.time()
        greedy_result = greedy_decoder.generate(
            prompt=prompt,
            max_length=80,
            temperature=1.0,
            repetition_penalty=1.1
        )
        greedy_time = time.time() - start_time
        
        print(f"⏱️ Generation time: {greedy_time:.3f}s")
        print(f"  Result: {greedy_result}")
        
        prompt_results['greedy'] = {
            'result': greedy_result,
            'time': greedy_time,
            'texts': [greedy_result]
        }
        
        # Sampling Decoding
        print(f"\n🎲 Nucleus Sampling (top-p=0.9):")
        sampling_results = []
        start_time = time.time()
        
        for _ in range(3):  # Generate 3 samples
            sample_result = sampling_decoder.generate(
                prompt=prompt,
                max_length=80,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1
            )
            sampling_results.append(sample_result)
        
        sampling_time = time.time() - start_time
        
        print(f"⏱️ Generation time: {sampling_time:.3f}s")
        for j, result in enumerate(sampling_results):
            print(f"  Sample {j+1}: {result}")
        
        prompt_results['sampling'] = {
            'results': sampling_results,
            'time': sampling_time,
            'texts': sampling_results
        }
        
        results[f'prompt_{i+1}'] = prompt_results
    
    # Overall analysis
    print(f"\n📊 COMPREHENSIVE ANALYSIS")
    print("=" * 50)
    
    # Speed comparison
    print(f"\n⚡ SPEED COMPARISON:")
    methods = ['beam_1', 'beam_3', 'beam_5', 'beam_10', 'greedy', 'sampling']
    method_names = ['Beam (k=1)', 'Beam (k=3)', 'Beam (k=5)', 'Beam (k=10)', 'Greedy', 'Sampling']
    
    for method, name in zip(methods, method_names):
        avg_time = np.mean([results[f'prompt_{i+1}'][method]['time'] 
                           for i in range(len(test_prompts)) 
                           if method in results[f'prompt_{i+1}']])
        print(f"  {name:<15}: {avg_time:.3f}s average")
    
    # Quality analysis
    print(f"\n📝 QUALITY ANALYSIS:")
    
    for method, name in zip(methods, method_names):
        all_texts = []
        for i in range(len(test_prompts)):
            if method in results[f'prompt_{i+1}']:
                all_texts.extend(results[f'prompt_{i+1}'][method]['texts'])
        
        if all_texts:
            quality_metrics = evaluate_text_quality(all_texts)
            print(f"\n  {name}:")
            print(f"    Average length: {quality_metrics['avg_length']:.1f} words")
            print(f"    Unique words: {quality_metrics['unique_words']}")
            print(f"    Repetition score: {quality_metrics['repetition_score']:.3f} (lower better)")
            print(f"    Diversity score: {quality_metrics['diversity_score']:.3f} (higher better)")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   🎯 BEAM SEARCH BENEFITS:")
    print(f"      - Higher beam width → better quality but slower generation")
    print(f"      - Length penalty prevents overly short sequences")
    print(f"      - Repetition penalty reduces repetitive text")
    print(f"      - Multiple candidates allow quality vs diversity trade-offs")
    print(f"   ")
    print(f"   ⚡ PERFORMANCE TRADE-OFFS:")
    print(f"      - Greedy: Fastest but least diverse")
    print(f"      - Beam (k=3-5): Good balance of quality and speed")
    print(f"      - Beam (k=10+): High quality but significant computational cost")
    print(f"      - Sampling: Diverse but unpredictable quality")
    print(f"   ")
    print(f"   🎨 CREATIVE APPLICATIONS:")
    print(f"      - Story completion: Beam search with length penalty")
    print(f"      - Dialogue generation: Lower beam width for faster response")
    print(f"      - Creative writing: Sampling for maximum diversity")
    print(f"      - Technical writing: Greedy for consistency")
    
    return results

def demonstrate_advanced_features():
    """Demonstrate advanced beam search features"""
    
    print(f"\n🔬 ADVANCED BEAM SEARCH FEATURES DEMONSTRATION")
    print("=" * 55)
    
    # Create a simple setup for demonstration
    training_stories = create_training_data()
    tokenizer = Tokenizer()
    tokenizer.build_vocabulary(training_stories[:10], min_freq=1)  # Smaller for demo
    model = train_simple_model(tokenizer, training_stories[:10], 64, 128)  # Smaller model
    
    beam_decoder = BeamSearchDecoder(model, tokenizer)
    
    test_prompt = "The scientist discovered a portal to another dimension where"
    
    # Demonstrate different parameter effects
    configs = [
        {"name": "Standard", "beam_width": 5, "length_penalty": 1.0, "repetition_penalty": 1.0},
        {"name": "Favor Longer", "beam_width": 5, "length_penalty": 1.5, "repetition_penalty": 1.0},
        {"name": "Reduce Repetition", "beam_width": 5, "length_penalty": 1.0, "repetition_penalty": 1.5},
        {"name": "High Diversity", "beam_width": 5, "length_penalty": 1.2, "repetition_penalty": 1.2, "diversity_penalty": 0.5},
    ]
    
    for config in configs:
        print(f"\n🔧 Configuration: {config['name']}")
        config_copy = config.copy()
        del config_copy['name']
        
        results = beam_decoder.generate(
            prompt=test_prompt,
            max_length=60,
            num_return_sequences=3,
            **config_copy
        )
        
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['text']}")
            print(f"     Score: {result['score']:.3f}")

if __name__ == "__main__":
    # Run comprehensive analysis
    print("🚀 Starting Advanced Text Generation Analysis...")
    
    results = run_comprehensive_text_generation()
    
    # Demonstrate advanced features
    demonstrate_advanced_features()
    
    print(f"\n🎯 BEAM SEARCH TEXT GENERATION MASTERY ACHIEVED!")
    print(f"   - Implemented sophisticated beam search decoder")
    print(f"   - Compared with greedy and sampling methods")
    print(f"   - Analyzed quality vs speed trade-offs")
    print(f"   - Demonstrated real-world text generation")
    print(f"   - Applied advanced features (length penalty, repetition penalty, diversity)")
    print(f"   - Built complete story completion system")