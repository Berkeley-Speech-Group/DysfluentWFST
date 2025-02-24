import torch
import k2
import cmudict
import json
from jiwer import wer
import math
import torch.nn.functional as F
import time

class WFSTdecoder:
    def __init__(self, device: str, phoneme_lexicon: list, is_ipa=False):
        self.device = torch.device(device)
        self.lexcion = phoneme_lexicon
        self.is_ipa = is_ipa
                
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
    
    def create_emit_graph(self, probNT):
        if probNT.ndim != 2:
            raise ValueError(f"probNT must be a 2D tensor, but got {probNT.ndim}D tensor.")
        lines = []
        for i in range(probNT.shape[0]):
            for j in range(probNT.shape[1]-1):
                lines.append(f"{i} {i+1} {j} {probNT[i, j]}")
        lines.append(f"{probNT.shape[0]} {probNT.shape[0]+1} {-1} {1}")
        lines.append(f"{probNT.shape[0]+1}")
        return '\n'.join(lines)
    
    def cmu2ipa(self, phoneme_seq, map='config/ipa2cmu.json'):
        map_dict = json.load(open(map))
        ipa_seq = []
        flag = False
        for phoneme in phoneme_seq:
            if phoneme == '<unk>':
                print(f"Phoneme {phoneme} not found in the CMU dictionary")
                continue
            flag = False
            for k, v in  map_dict.items():
                if phoneme in v:
                    flag = True
                    if ' ' in k:
                        k = k.split()
                        ipa_seq.extend(k)
                        break
                    ipa_seq.append(k)
                    break
            if not flag:
                raise ValueError(f"Phoneme {phoneme} not found in the map")
        return ipa_seq
    
    def get_phoneme_sequence(self, ref_text):
        phoneme_sequence = []
        ref_text = ref_text.lower()
        ref_text = ref_text.replace('.', '').replace(',', '').replace('?', '').replace('!', '')
        for word in ref_text.split():
            phonemes = cmudict.dict().get(word.lower(), ['<unk>'])[0]
            if phonemes == '<unk>':
                if word == 'quivers':
                    phonemes = 'K W IH V ER S'.split()
                else:
                    print(f"Word {word} not found in the CMU dictionary")
                    continue
            phonemes = [phoneme.rstrip('012') for phoneme in phonemes]
            phoneme_sequence.extend(phonemes)
        if self.is_ipa:
            ipa_sequence = self.cmu2ipa(phoneme_sequence)
        else:
            ipa_sequence = phoneme_sequence
        return ipa_sequence

    def get_phoneme_id(self, phoneme):
        if phoneme == '|' or phoneme == '-' or phoneme == '<pad>' or phoneme == '<s>' or phoneme == '</s>' or phoneme == '<unk>' or phoneme == 'SIL' or phoneme == 'SPN':
            return 0
        return self.lexcion.index(phoneme)
    
    def get_phoneme_ids(self, phoneme_sequence):
        return [self.get_phoneme_id(phoneme) for phoneme in phoneme_sequence]
    
    def create_fsa_graph(self, phonemes, beta, skip=False, back=True):
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
                        self.lexcion.append(mis_token)
                        mis_id = self.lexcion.index(mis_token)
                        lines.append(f"{i} {j} {0} {mis_id} {error_score * math.exp(-(i-j)**2/2) / 5000}")
                        continue
                else:
                    if alpha == 1:
                        continue
                    if j > i:
                        if skip:
                            if j - i > 3:
                                continue
                            mis_token = '<trans>'.join([str(i), str(j)])
                            self.lexcion.append(mis_token)
                            mis_id = self.lexcion.index(mis_token)
                            lines.append(f"{i} {j} {0} {mis_id} {error_score * math.exp(-(i-j)**2/2) / 100}")
                            continue
                    if j < i:
                        if back:
                            if i - j > 2:
                                continue
                            rep_token = '<trans>'.join([str(i), str(j)])
                            self.lexcion.append(rep_token)
                            rep_id = self.lexcion.index(rep_token)
                            lines.append(f"{i} {j} {0} {rep_id} {error_score * math.exp(-(i-j)**2/2) / 5000}")
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
                    "start_state": start,
                    "end_state": end,
                    "dysfluency_type": "repetition"
                })
            # Check for insertion (insertion occurs when start is earlier than any previous time)
            elif start < min_time:
                dysfluency_results.append({
                    "phoneme": phoneme,
                    "start_state": start,
                    "end_state": end,
                    "dysfluency_type": "insertion"
                })
            # Otherwise, it's a normal transition
            elif start > prev_end + 1:
                dysfluency_results.append({
                    "phoneme": "<del>",
                    "start_state": prev_end,
                    "end_state": start,
                    "dysfluency_type": "deletion"
                })
                dysfluency_results.append({
                    "phoneme": phoneme,
                    "start_state": start,
                    "end_state": end,
                    "dysfluency_type": "normal"
                })
            else:
                dysfluency_results.append({
                    "phoneme": phoneme,
                    "start_state": start,
                    "end_state": end,
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
    
    
    def decode(self, batch, beta, num_beam=2, back=True, skip=False):
        ids = batch["id"]  # List of IDs in the batch
        emissions = batch["tensor"]  # Batched emission tensors (padded)
        ref_texts = batch["ref_text"]  # List of reference texts
        lengths = batch["lengths"]  # Original lengths of each sequence in the batch

        results = []  # To store results for each sample
        
        # ctc graph (only needs to be created once for the batch)

        # Iterate over each sample in the batch
        for idx in range(len(ids)):
            sample_id = ids[idx]
            emission = emissions[idx, : lengths[idx]]  # Extract valid emission based on length
            print(emission.shape)
            emission = emission.to(self.device)
            ref_text = ref_texts[idx]

            # Create dense FSA for the current sample
            sample_probs_fsa = self.create_dense_fsa_vec(emission.unsqueeze(0), torch.tensor([lengths[idx]], dtype=torch.int32))
            sample_probs_fsa = sample_probs_fsa.to(self.device)

            # Reference FSA graph
            phoneme_sequence = self.get_phoneme_sequence(ref_text)
            phoneme_ids = self.get_phoneme_ids(phoneme_sequence)
            ref_fsa_str = self.create_fsa_graph(phoneme_ids, beta=beta, skip=skip, back=back)
            ref_fsa = k2.Fsa.from_str(ref_fsa_str, acceptor=False)
            ref_fsa = k2.arc_sort(ref_fsa)
            ref_fsa.to(self.device)
            
            ctc_fsa = self.ctc_topo(len(self.lexcion))
            ctc_fsa = k2.arc_sort(ctc_fsa)
            ctc_fsa.to(self.device)


            # Compose and intersect dense FSA
            composed = k2.compose(ctc_fsa, ref_fsa, treat_epsilons_specially=True)
            composed.to(self.device)
            lattices = k2.intersect_dense(composed, sample_probs_fsa, output_beam=num_beam)
            lattices.to(self.device)
            start_time = time.time()
            
            shortest = k2.shortest_path(lattices, use_double_scores=True)
            phoneme_list = [self.lexcion[i] for i in shortest[0].aux_labels[:-1]]

            # Deduplicate and filter phonemes
            phoneme_list = self._deduplicate_and_filter(phoneme_list)
            extract_phoneme_list = self.extract_phoneme_states(phoneme_list)
            dys_decode = self.detect_dysfluency(extract_phoneme_list)


            # Append results for the current sample
            results.append({
                "id": sample_id,
                "ref_phonemes": phoneme_sequence,
                "dys_detect": dys_decode,
                "decode_phonemes": [item["phoneme"] for item in dys_decode]
            })
            end_time = time.time()
            print(f"Time taken: {end_time - start_time}")

            # Clear CUDA memory for the current sample
                        
            del(sample_probs_fsa, ref_fsa, composed, lattices, shortest)
            torch.cuda.empty_cache()

        return results

