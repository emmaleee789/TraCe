import numpy as np

def w_star(t, m1=800, m2=500, s1=300, s2=100):
    # max time 1000
    r = np.ones_like(t) * 1.0
    r[t > m1] = np.exp(-((t[t > m1] - m1) ** 2) / (2 * s1 * s1))
    r[t < m2] = np.exp(-((t[t < m2] - m2) ** 2) / (2 * s2 * s2))
    return r

def precompute_prior(T=1000, min_t=200, max_t=800):
    ts = np.arange(T)
    prior = w_star(ts)[min_t:max_t]
    prior = prior / prior.sum()
    prior = prior[::-1].cumsum()[::-1]
    return prior, min_t

def w_star_bridge(t, m1=800, m2=500, s1=300, s2=50, phase_id=1):
    # # max time 1000
    # # Reflect t across the center (500) to achieve symmetry
    # t_reflected = 1000 - t
    # r = np.ones_like(t) * 1.0 # Use original t for shape
    # # Use t_reflected for the conditions and calculations
    # cond_gt_m1 = t_reflected > m1
    # cond_lt_m2 = t_reflected < m2
    # r[cond_gt_m1] = np.exp(-((t_reflected[cond_gt_m1] - m1) ** 2) / (2 * s1 * s1))
    # r[cond_lt_m2] = np.exp(-((t_reflected[cond_lt_m2] - m2) ** 2) / (2 * s2 * s2))
    # return r

    #===============================================

    # max time 1000
    # if phase_id == 1:
    #     r = np.ones_like(t) * 1.0
    #     r[t > 998] = np.exp(-((t[t > 998] - 998) ** 2) / (2 * s1 * s1))
    #     r[t < 300] = np.exp(-((t[t < 300] - 300) ** 2) / (2 * 100 * 100))
    #     return r
    # elif phase_id == 2:
    #     r = np.ones_like(t) * 1.0
    #     r[t > 700] = np.exp(-((t[t > 700] - 700) ** 2) / (2 * s2 * s2))
    #     r[t < 200] = np.exp(-((t[t < 200] - 200) ** 2) / (2 * 200 * 200))
    #     return r

    #===============================================

    r = np.ones_like(t) * 1.0
    r[t > 980] = np.exp(-((t[t > 980] - 980) ** 2) / (2 * s2 * s2))
    r[t < 20] = np.exp(-((t[t < 20] - 20) ** 2) / (2 * s1 * s1))

    #===============================================

    # # Create a symmetric weight function
    # # First half (t=0 to t=500) uses the pattern from file_context_0
    # # Second half (t>500) uses the pattern defined above
    # mid_point = 500
    # mask_first_half = t <= mid_point
    # mask_second_half = t > mid_point
    
    # r = np.ones_like(t) * 1.0
    # # Apply the first half pattern (from file_context_0)
    # r[mask_first_half & (t > 400)] = np.exp(((t[mask_first_half & (t > 400)] - 400) ** 2) / (2 * 200 * 200))
    # r[mask_first_half & (t < 200)] = np.exp(((t[mask_first_half & (t < 200)] - 200) ** 2) / (2 * s1 * s1))
    # # Apply the second half pattern (from file_context_1)
    # r[mask_second_half & (t > 800)] = np.exp(((t[mask_second_half & (t > 800)] - 800) ** 2) / (2 * s1 * s1))
    # r[mask_second_half & (t < 600)] = np.exp(((t[mask_second_half & (t < 600)] - 600) ** 2) / (2 * 200 * 200))


    return r


def precompute_prior_bridge(T=1000, min_t=200, max_t=800, phase_id=1):
    ts = np.arange(T)
    prior = w_star_bridge(ts, phase_id=phase_id)[min_t:max_t]
    prior = prior / prior.sum()
    prior = prior[::-1].cumsum()[::-1]
    return prior, min_t

def time_prioritize(step_ratio, time_prior, min_t=200):
    return np.abs(time_prior - step_ratio).argmin() + min_t