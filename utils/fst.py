import torch
import k2
import cmudict
import json
from jiwer import wer
import math
import torch.nn.functional as F
import time
import numpy as np
from .phn import PhonemeProcessor

class WFSTdecoder:
    def __init__(self, device: str, phoneme_lexicon: list, phn_type=False):
        self.device = torch.device(device)
        self.lexicon = phoneme_lexicon
        self.phn_type = phn_type
        # read from npy file

                
    def ctc_topo(self, num_phonemes: int):
        return k2.ctc_topo(max_token=num_phonemes, modified=False)
    
    def create_dense_fsa_vec(self, log_probs: torch.Tensor, lengths: torch.Tensor) -> k2.DenseFsaVec:
        """
        Create a DenseFsaVec from model outputs (log_probs) and sequence lengths.

        Args:
            log_probs (torch.Tensor): A tensor of shape (B, T, C), where
                - B: Batch size,
                - T: Number of time steps,
                - C: Number of classes (including blank).
            lengths (torch.Tensor): A tensor of shape (B,) containing the valid lengths
                (number of time steps) for each sample in the batch.

        Returns:
            k2.DenseFsaVec: A DenseFsaVec object that represents the dense FSA for each sequence.

        Raises:
            ValueError: If the input tensors do not have compatible dimensions.
        """
        # Validate the dimensions of log_probs
        if log_probs.ndim != 3:
            raise ValueError(f"log_probs must be a 3D tensor of shape (B, T, C), but got {log_probs.ndim}D tensor.")

        B, T, C = log_probs.shape

        if lengths.shape[0] != B:
            raise ValueError(f"The size of lengths must match the batch size, but got {lengths.shape[0]} and {B}.")
        lengths = lengths.to(dtype=torch.int32)
        
        
        log_probs = F.log_softmax(log_probs, dim=-1)

        supervision_segments = []
        for i in range(B):
            supervision_segments.append([i, 0, lengths[i].item()])
        supervision_segments = torch.tensor(supervision_segments, dtype=torch.int32)

        dense_fsa_vec = k2.DenseFsaVec(log_probs, supervision_segments)
        return dense_fsa_vec
    
    def get_phoneme_sequence(self, ref_text):
        processor = PhonemeProcessor()
        return processor.get_phoneme_sequence(ref_text, self.phn_type)

    def get_phoneme_id(self, phoneme):
        if phoneme == '|' or phoneme == '-' or phoneme == '<pad>' or phoneme == '<s>' or phoneme == '</s>' or phoneme == '<unk>' or phoneme == 'SIL' or phoneme == 'SPN':
            return 0
        return self.lexicon.index(phoneme)
    
    def get_phoneme_ids(self, phoneme_sequence):
        return [self.get_phoneme_id(phoneme) for phoneme in phoneme_sequence]
    
    def create_fsa_graph(self, phonemes, beta, skip=False, back=True, sub=True):
        phn_processor = PhonemeProcessor()
        alpha = 1 - 10**(-beta)
        error_score = (1-alpha)
        lines = []
        for i, phone in enumerate(phonemes):
            for j in range(len(phonemes)+1):
                if i == j:
                    continue
                if j == i+1:
                    lines.append(f"{i} {j} {phone} {phone} {alpha}")
                    if skip:
                        mis_token = '<trans>'.join([str(i), str(j)])
                        self.lexicon.append(mis_token)
                        mis_id = self.lexicon.index(mis_token)
                        lines.append(f"{i} {j} {0} {mis_id} {error_score * math.exp(-(i-j)**2/2)}")
                        continue
                    if sub:
                        # phoneme id 2 phoneme
                        phone_text = self.lexicon[phone]
                        # if self.is_ipa:
                        #     phone_text = self.ipa2cmu(phone_text)
                        # phoneme_id_sim = self.phn2idx[phone_text]
                        # select top3 similar phonemes's id
                        sim_phonemes = phn_processor.get_similar_phonemes(phone_text, 3, self.phn_type)
                        # create unique sim_tokens, eg: EH -> <sub>EH
                        sim_tokens = ['<sub>' + sp for sp in sim_phonemes if sp != phone_text]
                        for sim_token in sim_tokens:
                            if sim_token not in self.lexicon:
                                self.lexicon.append(sim_token)
                        # get sim phoneme ids from sim_tokens
                        sim_phoneme_ids = [self.lexicon.index(st) for st in sim_tokens]

                        for sim_id in sim_phoneme_ids:
                            # sim_phoneme = list(self.phn2idx.keys())[sim_id]
                            # # print(f"sim_phoneme: {sim_phoneme}")
                            # if self.is_ipa:
                            #     sim_phoneme = self.cmu2ipa([sim_phoneme])[0]
                            # try:
                            #     sim_phoneme_id = self.get_phoneme_id(sim_phoneme)
                            # except ValueError:
                            #     print(f"Phoneme {sim_phoneme} not found in the lexicon")
                            #     continue
                            # sub_token = '<trans>'.join([str(i), str(j)])
                            # self.lexicon.append(sub_token)
                            # sub_id = self.lexicon.index(sub_token)
                            lines.append(f"{i} {j} {phone} {sim_id} {error_score/10000}")
                else:
                    if alpha == 1:
                        continue
                    if j > i and skip:
                        if j - i > 3:
                            continue
                        mis_token = '<trans>'.join([str(i), str(j)])
                        self.lexicon.append(mis_token)
                        mis_id = self.lexicon.index(mis_token)
                        lines.append(f"{i} {j} {0} {mis_id} {error_score * math.exp(-(i-j)**2/2)}")
                        continue
                    if j < i and back:
                        if i - j > 2:
                            continue
                        rep_token = '<trans>'.join([str(i), str(j)])
                        self.lexicon.append(rep_token)
                        rep_id = self.lexicon.index(rep_token)
                        lines.append(f"{i} {j} {0} {rep_id} {error_score * math.exp(-(i-j)**2/2)}")
                        continue
                    
        lines.append(f"{len(phonemes)} {len(phonemes)+1} {-1} {-1} {0}")
        lines.append(f"{len(phonemes)+1}")
        return '\n'.join(lines)

    def extract_phoneme_states(self, transition_list):
        merged_list = []
        current_merge = None
        
        for item in transition_list:
            if "<trans>" in item:
                if current_merge is None:
                    current_merge = item
                else:
                    current_merge = current_merge.split("<trans>")[0] + "<trans>" + item.split("<trans>")[-1]
            else:
                if current_merge is not None:
                    merged_list.append(current_merge)
                    current_merge = None
                merged_list.append(item)
        
        # Handle case where the last item was a merge
        if current_merge is not None:
            merged_list.append(current_merge)
        
        return merged_list
    
    def detect_dysfluency(self, phoneme_seq):
        dysfluency_results = []
        state_history = set()
        prev_end = -1
        
        
        clean_states = []
        current_state = 0
        
        for elem in phoneme_seq:
            if '<trans>' in elem:
                _, j = elem.split('<trans>')
                current_state = int(j)
            else:
                start = current_state
                end = start + 1
                clean_states.append((start, end, elem))
                current_state = end

        # Detect dysfluency
        for item in clean_states:
            start, end, phoneme = item
            # get the minimum time in state_history: [(1, 2, 'ɛ'), ...]
            min_time = min(state_history) if state_history else -1            
            if start in state_history:
                dysfluency_results.append({
                    "phoneme": phoneme,
                    # "start_state": start,
                    # "end_state": end,
                    "dysfluency_type": "repetition"
                })
            # Check for insertion (insertion occurs when start is earlier than any previous time)
            elif start < min_time:
                dysfluency_results.append({
                    "phoneme": phoneme,
                    # "start_state": start,
                    # "end_state": end,
                    "dysfluency_type": "insertion"
                })
            # Otherwise, it's a normal transition
            elif start > prev_end + 1:
                dysfluency_results.append({
                    "phoneme": "<del>",
                    # "start_state": prev_end,
                    # "end_state": start,
                    "dysfluency_type": "deletion"
                })
                dysfluency_results.append({
                    "phoneme": phoneme,
                    # "start_state": start,
                    # "end_state": end,
                    "dysfluency_type": "normal"
                })
            else:
                if "<sub>" in phoneme:
                    dysfluency_results.append({
                        "phoneme": phoneme.replace("<sub>", ""),
                        # "start_state": start,
                        # "end_state": end,
                        "dysfluency_type": "substitution"
                    })
                else:
                    dysfluency_results.append({
                        "phoneme": phoneme,
                        # "start_state": start,
                        # "end_state": end,
                        "dysfluency_type": "normal"
                    })
            state_history.add(start)
            prev_end = end

        return dysfluency_results
    
    def _deduplicate_and_filter(self, phoneme_list):
        """Deduplicate consecutive phonemes and filter out unwanted tokens."""
        filtered_list = []
        prev_label = None
        for phoneme in phoneme_list:
            if phoneme != prev_label and phoneme not in ['|', '-', "<pad>", "<s>", "</s>", "<unk>", 'STL', 'SPN', '<blank>', '<b>']:
                filtered_list.append(phoneme)
            prev_label = phoneme
        return filtered_list
    
    

    def _build_lattice(self, emission, length, ref_text,
                       beta, back, skip, num_beam):
        """Create the k2 lattice for one utterance and return it together
        with the reference phoneme sequence."""
        emission = emission.to(self.device)

        # 1. Dense FSA from model post‑eriors
        dense_fsa = self.create_dense_fsa_vec(
            emission.unsqueeze(0),
            torch.tensor([length], dtype=torch.int32)
        ).to(self.device)

        # 2. Reference text → phoneme IDs → FSA
        phoneme_sequence = self.get_phoneme_sequence(ref_text)
        phoneme_ids = self.get_phoneme_ids(phoneme_sequence)
        ref_fsa_str = self.create_fsa_graph(
            phoneme_ids, beta=beta, skip=skip, back=back
        )
        ref_fsa = k2.Fsa.from_str(ref_fsa_str, acceptor=False)
        ref_fsa = k2.arc_sort(ref_fsa).to(self.device)

        # 3. CTC topology
        ctc_fsa = self.ctc_topo(len(self.lexicon))
        ctc_fsa = k2.arc_sort(ctc_fsa).to(self.device)

        # 4. Compose & intersect to obtain the lattice
        composed = k2.compose(
            ctc_fsa.to("cpu"),
            ref_fsa.to("cpu"),
            treat_epsilons_specially=True
        ).to(self.device)
        lattice = k2.intersect_dense(
            composed, dense_fsa, output_beam=num_beam
        ).to(self.device)

        return lattice, phoneme_sequence

    # def _compute_loss(self, lattice):
    #     loss = lattice.get_tot_scores(
    #         log_semiring=True, use_double_scores=True
    #     )
    #     return -loss.mean()

    def _decode_lattice(self, lattice):
        """Return dysfluency annotations derived from the shortest path."""
        shortest = k2.shortest_path(lattice, use_double_scores=True)
        # phoneme_seq = [self.lexicon[i] for i in shortest[0].aux_labels[:-1]]

        # phoneme_seq = self._deduplicate_and_filter(phoneme_seq)
        # state_seq   = self.extract_phoneme_states(phoneme_seq)
        # return self.detect_dysfluency(state_seq)
        return shortest

    def decode(self, batch, beta, num_beam=15, back=True, skip=False):
        """
        Decode a batch of sequences using the WFST decoder.
        Args:
            batch (dict): A dictionary containing the following
                - "id": List of IDs in the batch.
                - "tensor": Batched emission tensors (padded).
                - "ref_text": List of reference texts.
                - "lengths": Original lengths of each sequence in the batch.
            beta (float): Parameter for the FSA graph.
            num_beam (int): Number of beams for decoding.
            back (bool): Whether to allow back transitions in the FSA graph.
            skip (bool): Whether to allow skip transitions in the FSA graph.
            train (bool): Whether the model is in training mode.
        Returns:
            results (list): A list of dictionaries containing the following
                - "id": Sample ID.
                - "ref_phonemes": Reference phoneme sequence.
                - "dys_detect": List of detected dysfluencies.
                - "decode_phonemes": List of decoded phonemes.
        """
        ids       = batch["id"]
        emissions = batch["tensor"]
        ref_texts = batch["ref_text"]
        lengths   = batch["lengths"]

        results = []
        for idx, sample_id in enumerate(ids):
            emission   = emissions[idx, : lengths[idx]]
            ref_text   = ref_texts[idx]

            lattice, ref_phonemes = self._build_lattice(
                emission, lengths[idx], ref_text,
                beta, back, skip, num_beam
            )

            raw_lattice = self._decode_lattice(lattice)
            raw_phns = [self.lexicon[i] for i in raw_lattice[0].aux_labels[:-1]]
            raw_phns = self._deduplicate_and_filter(raw_phns)
            state_seq = self.extract_phoneme_states(raw_phns)
            dys_info = self.detect_dysfluency(state_seq)
            results.append({
                "id": sample_id,
                "ref_phonemes": ref_phonemes,
                "dys_detect":  dys_info,
                "decode_phonemes": [item["phoneme"] for item in dys_info],
                "lattice": raw_lattice
            })

            del lattice
            torch.cuda.empty_cache()

        return results

