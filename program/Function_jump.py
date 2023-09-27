import numpy as np
import pandas as pd
from scipy.stats import norm
import os

jump_path = 'C:/Users/86156/Desktop/option/data/jump_data/'


def compute_cv_kh(cv_list, h, M):
    CV_kh = []
    for k in range(1, M + 1):
        t_start = (k - 1) * h
        t_end = k * h
        cv_sum = np.sum(cv_list[t_start:t_end])
        CV_kh.append(cv_sum / (h ** 0.5))

    return CV_kh


def compute_intensity_kh(j_list, h, M):
    intensity_kh = []
    for k in range(1, M + 1):
        t_start = (k - 1) * h
        t_end = k * h
        intensity_sum = np.sum(np.array(j_list[t_start:t_end]) != 0)
        intensity_kh.append(intensity_sum)

    return intensity_kh


def calculate_j_ti(ret_matrix, Z_score, alpha):
    J_ti = []
    for t, Z_t in enumerate(Z_score):
        jumps = []
        for i, ret in enumerate(ret_matrix[t]):
            if i > 0 and Z_t > norm.ppf(1 - alpha):
                jump = (ret_matrix[t, i] ** 2 - 0.5 * (np.abs(ret_matrix[t, i - 1]) + np.abs(ret_matrix[t, i])))
            else:
                jump = 0
            jumps.append(jump)
        J_ti.append(jumps)

    return J_ti


def calculate_size_kh(J_ti, h, M, intensity_kh):
    size_kh = []
    for k in range(1, M + 1):
        t_start = (k - 1) * h
        t_end = k * h
        abs_jump_sum = 0

        for jt in J_ti[t_start:t_end]:
            for jt_i in jt:
                if np.abs(jt_i) > 0:
                    abs_jump_sum += np.abs(jt_i)

        size_k = abs_jump_sum / intensity_kh[k - 1] if intensity_kh[k - 1] > 0 else 0
        size_kh.append(size_k)

    return size_kh


def calculate_sizevol_kh(J_ti, h, M, intensity_kh, size_kh):
    sizevol_kh = []
    for k in range(1, M + 1):
        t_start = (k - 1) * h
        t_end = k * h
        squared_diff_sum = 0

        for t, jt in enumerate(J_ti[t_start:t_end], start=t_start):
            for jt_i in jt:
                if np.abs(jt_i) > 0:
                    squared_diff = (np.abs(jt_i) - size_kh[k - 1]) ** 2
                    squared_diff_sum += squared_diff

        sizevol_k = squared_diff_sum / intensity_kh[k - 1] if intensity_kh[k - 1] > 0 else 0
        sizevol_kh.append(sizevol_k)

    return sizevol_kh


def find_jumps(ret_matrix, alpha):
    def calculate_RV_BV_TQ(rt):
        n = len(rt)
        RV_t = np.sum(rt ** 2)
        BV_t = 0.5 * np.pi * np.sum(np.abs(rt[:-1]) * np.abs(rt[1:]))
        TQ_t = (1 / n) * 1.7432 * np.sum(np.abs(rt[:-2]) ** (3 / 4) * np.abs(rt[1:-1]) ** (3 / 4) * np.abs(rt[2:]) ** (3 / 4))
        return RV_t, BV_t, TQ_t

    def Z_statistic(rt, TQ_t, RV_t, BV_t):
        n = len(rt)
        Z_t = ((n ** 0.5) * (RV_t - BV_t) * (RV_t ** (-1))) / (0.6087 * max(1, TQ_t * (BV_t ** (-2)))) ** 0.5
        return Z_t

    jumps = []

    for rt in ret_matrix:
        jump_indices = []
        while True:
            RV_t, BV_t, TQ_t = calculate_RV_BV_TQ(rt)
            Z_t = Z_statistic(rt, TQ_t, RV_t, BV_t)
            alpha_quantile = norm.ppf(1 - alpha)

            if Z_t <= alpha_quantile:
                break

            r_med = np.median(np.delete(rt, jump_indices))
            rt_no_jump = np.where(np.isin(np.arange(len(rt)), jump_indices), r_med, rt)
            Z_no_jump = [Z_statistic(np.where(rt_no_jump == r, r_med, rt_no_jump), TQ_t, RV_t, BV_t) for r in rt_no_jump]

            max_diff = -1
            max_diff_index = -1
            for i, (z_orig, z_no_jump) in enumerate(zip([Z_t] * len(Z_no_jump), Z_no_jump)):
                diff = z_orig - z_no_jump
                if diff > max_diff:
                    max_diff = diff
                    max_diff_index = i

            jump_indices.append(max_diff_index)

        J_t = sum([rt[i] for i in jump_indices])
        print(J_t)
        jumps.append(J_t)
        print(1)

    return jumps
