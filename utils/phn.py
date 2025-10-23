# Phoneme processing utilities
# import eng_to_ipa as ipa
import cmudict
import string
import inflect
import numpy as np
import json
# import ipapy.ipastring
from typing import List, Union
import os
from g2p_en import G2p

def remove_stress(phoneme: str) -> str:
    import re
    """
    Remove stress from a phoneme.
    """
    return re.sub(r'\d', '', phoneme)


class PhonemeProcessor:
    """
    A comprehensive phoneme processing class that handles text-to-phoneme conversion,
    phoneme similarity computation, and conversion between IPA and CMU phoneme representations.
    """
    
    def __init__(self, ipa2cmu_map_path: str = '/home/chenxu/vscode_workspace/asr-llm-wfst/DysfluentWFST/config/ipa2cmu.json', 
                 sim_matrix_path: str = '/home/chenxu/vscode_workspace/asr-llm-wfst/DysfluentWFST/utils/similarity.npy',
                 metadata_path: str = '/home/chenxu/vscode_workspace/asr-llm-wfst/DysfluentWFST/utils/metadata.json'):
        """
        Initialize the PhonemeProcessor.
        
        Args:
            ipa2cmu_map_path: Path to the IPA to CMU mapping JSON file
            sim_matrix_path: Path to the phoneme similarity matrix numpy file
            metadata_path: Path to the metadata JSON file containing phoneme labels
        """
        self.ipa2cmu_map_path = ipa2cmu_map_path
        self.sim_matrix_path = sim_matrix_path
        self.metadata_path = metadata_path
        self.inflect_engine = inflect.engine()
        self.g2p = G2p()
        
        # Cache for loaded data
        self._ipa2cmu_map = None
        self._sim_matrix = None
        self._cmu_dict = None
        self._metadata = None
        self._phn2idx = None
    
    @property
    def metadata(self):
        """Lazy load metadata."""
        if self._metadata is None:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self._metadata = json.load(f)
            else:
                raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        return self._metadata
    
    @property
    def phn2idx(self):
        """Lazy load phoneme to index mapping from metadata."""
        if self._phn2idx is None:
            phoneme_labels = self.metadata.get('phoneme_labels', [])
            # Convert to uppercase to match CMU format
            self._phn2idx = {phn.upper(): idx for idx, phn in enumerate(phoneme_labels)}
        return self._phn2idx
    
    @property
    def ipa2cmu_map(self):
        """Lazy load IPA to CMU mapping."""
        if self._ipa2cmu_map is None:
            if os.path.exists(self.ipa2cmu_map_path):
                with open(self.ipa2cmu_map_path, 'r') as f:
                    self._ipa2cmu_map = json.load(f)
            else:
                raise FileNotFoundError(f"IPA to CMU mapping file not found: {self.ipa2cmu_map_path}")
        return self._ipa2cmu_map
    
    @property
    def sim_matrix(self):
        """Lazy load similarity matrix."""
        if self._sim_matrix is None:
            if os.path.exists(self.sim_matrix_path):
                self._sim_matrix = np.load(self.sim_matrix_path)
            else:
                raise FileNotFoundError(f"Similarity matrix file not found: {self.sim_matrix_path}")
        return self._sim_matrix
    
    @property
    def cmu_dict(self):
        """Lazy load CMU dictionary."""
        if self._cmu_dict is None:
            self._cmu_dict = cmudict.dict()
        return self._cmu_dict
    
    # @staticmethod
    # def is_ipa(char: str) -> bool:
    #     """
    #     Check if a character is a valid IPA character.
        
    #     Args:
    #         char: Single character to check
            
    #     Returns:
    #         True if character is valid IPA, False otherwise
    #     """
    #     if len(char) != 1:
    #         raise ValueError("Input must be a single character.")
    #     return ipapy.ipastring.is_valid_ipa(char)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for phoneme processing.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        text = text.lower()
        new_text = ''
        temp = ''
        
        for char in text:
            if char.isdigit():
                temp += char
            else:
                if temp != '':
                    new_text += self.inflect_engine.number_to_words(temp) + ' '
                    temp = ''
                new_text += char
        
        if temp != '':
            new_text += self.inflect_engine.number_to_words(temp) + ' '
        
        text = new_text.strip()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text
    
    def get_phoneme_sequence(self, text: str, phn_type: str) -> List[str]:
        """
        Convert text to phoneme sequence.
        
        Args:
            text: Input text
            phn_type: Phoneme type ('ipa' or 'cmu')
            
        Returns:
            List of phonemes
        """
        lexicon = json.load(open('/home/chenxu/vscode_workspace/asr-llm-wfst/DysfluentWFST/config/vocab.json', 'r'))
        lexicon = [k.upper() for k, v in lexicon.items()]
        if phn_type not in ['ipa', 'cmu']:
            raise ValueError("Unsupported phoneme type. Use 'ipa' or 'cmu'.")
        
        # Clean text
        text = self.clean_text(text)
        
        if phn_type == 'ipa':
            phonemes = self.g2p(text)
            phonemes = [remove_stress(p) for p in phonemes]
            phonemes = [p for p in phonemes if p != ' ']
            for phn in phonemes:
                if phn not in lexicon:
                    print(f"Phoneme {phn} not found in lexicon")
            phonemes = self.cmu2ipa(phonemes)
            # Remove spaces
            
            
        elif phn_type == 'cmu':
            phonemes = self.g2p(text)
            phonemes = [remove_stress(p) for p in phonemes]
            # Remove spaces
            phonemes = [p for p in phonemes if p != ' ' and p in lexicon]
        return phonemes
    
    def _get_sim_phoneme_cmu(self, phn: str, num: int) -> List[str]:
        """
        Get similar CMU phonemes based on similarity matrix.
        
        Args:
            phn: CMU phoneme (uppercase)
            num: Number of similar phonemes to return
            
        Returns:
            List of similar phonemes
        """
        # Convert to uppercase to match the phoneme mapping
        phn = phn.upper()
        
        if phn not in self.phn2idx:
            raise ValueError(f"Phoneme '{phn}' not found in phoneme list, please use CMU phonemes.")
        
        # Return the top num similar phonemes based on similarity matrix
        phn_idx = self.phn2idx[phn]
        # Sort by similarity (descending order) and get top num+1 (including itself)
        sim_indices = np.argsort(-self.sim_matrix[phn_idx])[:num+1]
        
        # Get phoneme names from indices, excluding the input phoneme itself
        idx2phn = {idx: phn for phn, idx in self.phn2idx.items()}
        sim_phns = [idx2phn[i] for i in sim_indices if i != phn_idx]
        
        # Return only the requested number of similar phonemes
        return sim_phns[:num]
    
    def ipa2cmu(self, phoneme: str) -> str:
        """
        Convert IPA phoneme to CMU phoneme.
        
        Args:
            phoneme: IPA phoneme
            
        Returns:
            CMU phoneme
        """
        if phoneme in self.ipa2cmu_map:
            cmu_value = self.ipa2cmu_map[phoneme].split()[0]
            return cmu_value
        else:
            raise ValueError(f"Phoneme {phoneme} not found in the map")
    
    def cmu2ipa(self, phoneme_seq: List[str]) -> List[str]:
        """
        Convert CMU phoneme sequence to IPA phoneme sequence.
        
        Args:
            phoneme_seq: List of CMU phonemes
            
        Returns:
            List of IPA phonemes
        """
        ipa_seq = []
        
        for phoneme in phoneme_seq:
            if phoneme == '<unk>':
                print(f"Phoneme {phoneme} not found in the CMU dictionary")
                continue
            
            found = False
            for k, v in self.ipa2cmu_map.items():
                if phoneme in v:
                    found = True
                    if ' ' in k:
                        k = k.split()
                        ipa_seq.extend(k)
                        break
                    ipa_seq.append(k)
                    break
            
            if not found:
                raise ValueError(f"Phoneme {phoneme} not found in the map")
        
        return ipa_seq
    
    def get_similar_phonemes(self, phn: str, num: int, phn_type: str) -> List[str]:
        """
        Get similar phonemes for a given phoneme.
        
        Args:
            phn: Input phoneme
            num: Number of similar phonemes to return
            phn_type: Phoneme type ('ipa' or 'cmu')
            
        Returns:
            List of similar phonemes
        """
        if phn_type == 'cmu':
            return self._get_sim_phoneme_cmu(phn, num)
        else:
            cmu_phn = self.ipa2cmu(phn)
            sim_cmu_phns = self._get_sim_phoneme_cmu(cmu_phn, num)
            return self.cmu2ipa(sim_cmu_phns)


# Legacy function wrappers for backward compatibility
# def is_ipa(char: str) -> bool:
#     """Legacy wrapper for PhonemeProcessor.is_ipa()"""
#     return PhonemeProcessor.is_ipa(char)

def text_cleaner(text: str) -> str:
    """Legacy wrapper for PhonemeProcessor.clean_text()"""
    processor = PhonemeProcessor()
    return processor.clean_text(text)

def get_phoneme_sequence(text: str, phn: str) -> List[str]:
    """Legacy wrapper for PhonemeProcessor.get_phoneme_sequence()"""
    processor = PhonemeProcessor()
    return processor.get_phoneme_sequence(text, phn)

def _get_sim_phoneme_cmu(phn: str, num: int) -> List[str]:
    """Legacy wrapper for PhonemeProcessor._get_sim_phoneme_cmu()"""
    processor = PhonemeProcessor()
    return processor._get_sim_phoneme_cmu(phn, num)

def ipa2cmu(phoneme: str, map: str = 'config/ipa2cmu.json') -> str:
    """Legacy wrapper for PhonemeProcessor.ipa2cmu()"""
    processor = PhonemeProcessor(ipa2cmu_map_path=map)
    return processor.ipa2cmu(phoneme)

def cmu2ipa(phoneme_seq: List[str], map: str = 'config/ipa2cmu.json') -> List[str]:
    """Legacy wrapper for PhonemeProcessor.cmu2ipa()"""
    processor = PhonemeProcessor(ipa2cmu_map_path=map)
    return processor.cmu2ipa(phoneme_seq)

def get_sim_phoneme(phn: str, num: int, phn_type: str) -> List[str]:
    """Legacy wrapper for PhonemeProcessor.get_similar_phonemes()"""
    processor = PhonemeProcessor()
    return processor.get_similar_phonemes(phn, num, phn_type)

if __name__ == "__main__":
    # Example usage of the PhonemeProcessor class
    processor = PhonemeProcessor()
    # ['clean_text', 'cmu2ipa', 'get_phoneme_sequence', 'get_similar_phonemes', 'ipa2cmu']
    # print all the public methods of the class
    print([method for method in dir(processor) if callable(getattr(processor, method)) and not method.startswith("_")])
