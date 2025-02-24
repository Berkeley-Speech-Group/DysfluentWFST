import numpy as np

class W_PER:
    def __init__(self):
        self.sim_matrix = np.load('./rule_sim_matrix.npy')
        self.phn2idx = {
                "|": 0, "OW": 1, "UW": 2, "EY": 3, "AW": 4, "AH": 5, "AO": 6, "AY": 7, "EH": 8, "K": 9,
                "NG": 10, "F": 11, "JH": 12, "M": 13, "CH": 14, "IH": 15, "UH": 16, "HH": 17, "L": 18,
                "AA": 19, "R": 20, "TH": 21, "AE": 22, "D": 23, "Z": 24, "OY": 25, "DH": 26, "IY": 27, "B": 28, "W": 29, "S": 30,
                "T": 31, "SH": 32, "ZH": 33, "ER": 34, "V": 35, "Y": 36, "N": 37, "G": 38, "P": 39, "-": 40
            }
    
    def weight_per(self, GT_list, hypo_phn_list):
        sim_matrix = self.sim_matrix
        phn2idx = self.phn2idx
        n = len(GT_list)
        m = len(hypo_phn_list)
        
        # init dp
        dp = np.zeros((n + 1, m + 1))
        insertion_cost = 1
        deletion_cost = 1
        
        for i in range(1, n + 1):
            dp[i][0] = i * deletion_cost
        for j in range(1, m + 1):
            dp[0][j] = j * insertion_cost
        

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                ref_idx = phn2idx[GT_list[i - 1]]
                hyp_idx = phn2idx[hypo_phn_list[j - 1]]
                substitution_cost = 1 - sim_matrix[ref_idx][hyp_idx]
                
                dp[i][j] = min(
                    dp[i - 1][j - 1] + substitution_cost, 
                    dp[i - 1][j] + deletion_cost,         
                    dp[i][j - 1] + insertion_cost         
                )
        
        return dp[n][m] / n