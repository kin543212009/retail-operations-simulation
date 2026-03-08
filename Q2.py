import math
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 超級市場/餐廳排隊及營收系統
# ==============================================================================
# 1. 模型核心參數配置（可調整模擬次數 NUM_RUNS）
# ==============================================================================
MINUTE=60 #DON'T CHANGE
# 收銀台配置
MAX_SERVERS = 2                  # 總收銀台數(可修改為任意正整數)
QUEUE_CAP_PER_SERVER = 8         # 單隊列最大容量
OPEN_THRESHOLD = 4                # 開台閾值：所有已開啟收銀台隊列長度≥此值則開新台
MIN_SERVERS = 1                   # 最小保留收銀台數（關台時至少保留）
IDLE_CLOSE_THRESHOLD = 1.0        # 閒置超過此時間（分鐘）則關閉收銀台

# 時間配置
SIM_START = 0 * MINUTE              # 模擬開始
PEAK_START = 7.5 * MINUTE           # 高峰開始
PEAK_STEADY_START = 12.5 * MINUTE    # 高峰平穩期開始
PEAK_STEADY_END = 17.5 * MINUTE      # 高峰平穩期結束
PEAK_END = 19 * MINUTE             # 高峰結束
SIM_END =  24* MINUTE                # 模擬結束

# 到達速率配置
EXPECTED_BASE_LAMBDA = 1           # 非高峰到達速率（人/分鐘）          
PEAK_MULTIPLIER = 3                # 高峰最大倍數

# 服務時間配置（指數分佈）
SERVICE_MEAN_THEORY = 25 / MINUTE    #(秒）

# 消費金額配置（對數正態分佈）
REVENUE_MEAN = 40                # 單客平均消費：60元
REVENUE_VAR = 15                 # 消費金額方差：20

# 方差縮減配置
PRE_RUNS = 100                    # 預模擬次數
NUM_RUNS = 500                   # 正式模擬次數
SEED_OFFSET = 99999              # 預模擬種子偏移量

# ==============================================================================
# 2. 輔助函數：自動計算對數正態分佈參數
# ==============================================================================
def calculate_lognormal_params(mean, var):
    """根據業務均值/方差推導對數正態的μ和σ"""
    sigma_sq = math.log(1 + var / (mean ** 2))
    sigma = math.sqrt(sigma_sq)
    mu = math.log(mean) - sigma_sq / 2
    return mu, sigma

LOGNORMAL_MU, LOGNORMAL_SIGMA = calculate_lognormal_params(REVENUE_MEAN, REVENUE_VAR)
print(f"\n對數正態參數：μ={LOGNORMAL_MU:.4f}，σ={LOGNORMAL_SIGMA:.4f}")

# ==============================================================================
# 3. 隨機數生成器
# ==============================================================================
class RandomGenerator:
    """獨立隨機數生成器，保證對偶變量同步"""
    def __init__(self, seed, use_dual=False):
        self.seed = seed
        self.use_dual = use_dual
        self.rng = random.Random(seed)  # 獨立隨機流
    
    def uniform(self):
        """生成[0,1)均勻隨機數，支持對偶變量（1-U）"""
        u = self.rng.random()
        return 1 - u if self.use_dual else u
    
    def uniform_no_dual(self):
        """生成[0,1)均勻隨機數，強制不使用對偶變量（用於Thinning接受判定）"""
        return self.rng.random()
    
    def exponential(self, lam):
        """逆變換法生成指數分佈（到達間隔/服務時間）"""
        u = self.uniform()
        return -math.log(1 - u) / lam
    
    def lognormal(self, mu, sigma):
        """
        修復對偶變量實現：基於Box-Muller的z取反，而非修改u1/u2
        逆變換法生成對數正態分佈（消費金額）
        """
        # 標準Box-Muller生成標準正態分佈
        u1 = self.rng.uniform(1e-10, 1.0)
        u2 = self.rng.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        
        # 對偶變量：z取反（核心修復點）
        if self.use_dual:
            z = -z
        
        # 還原為對數正態分佈
        return math.exp(mu + sigma * z)

# ==============================================================================
# 4. 核心：正確的非齊次泊松過程（NHPP）生成器（Thinning算法）- 优化高峰客流建模
# ==============================================================================
def generate_nhpp_arrival(current_time, rng, lambda_non_peak, lambda_max_peak):
    """
    用Thinning算法生成NHPP的下一個到達時間
    """
    t = current_time
    while True:
        inter_arrival = rng.exponential(lambda_max_peak)
        t += inter_arrival
        
        if t >= SIM_END:
            return t
        
        # 三段式高峰客流建模（上升→平穩→下降）
        if PEAK_START <= t <= PEAK_END:
            peak_total_duration = PEAK_END - PEAK_START
            steady_start = PEAK_STEADY_START - PEAK_START  
            steady_end = PEAK_STEADY_END - PEAK_START      
            t_peak = t - PEAK_START                        
            
            if t_peak < steady_start:
                progress = t_peak / steady_start
                lambda_t = lambda_non_peak + (lambda_max_peak - lambda_non_peak) * progress
            elif steady_start <= t_peak < steady_end:
                lambda_t = lambda_max_peak
            else:
                fall_duration = peak_total_duration - steady_end
                progress = (t_peak - steady_end) / fall_duration
                lambda_t = lambda_max_peak - (lambda_max_peak - lambda_non_peak) * progress
        else:
            lambda_t = lambda_non_peak
        
        lambda_t = max(lambda_t, 0.1)
        
        u = rng.uniform_no_dual()
        if u <= lambda_t / lambda_max_peak:
            return t

# ==============================================================================
# 5. 事件模擬核心（隊列分配+收銀台狀態邊界，關閉閒置收銀台逻辑）
# ==============================================================================
def simulate_queue(seed, use_dual=False, process_remaining_queue=True):
    """單次離散事件模擬"""
    rng = RandomGenerator(seed, use_dual)
    
    # 直接使用基礎設定的期望值
    daily_lambda_non_peak = EXPECTED_BASE_LAMBDA
    daily_lambda_max_peak = EXPECTED_BASE_LAMBDA * PEAK_MULTIPLIER
    
    t = SIM_START
    active_servers = 1  
    server_busy = [False] * MAX_SERVERS
    server_depart_time = [float('inf')] * MAX_SERVERS
    queues = [[] for _ in range(MAX_SERVERS)]
    server_idle_start = [float('inf')] * MAX_SERVERS  
    
    total_hours = int((SIM_END - SIM_START) / 60)
    hourly_arrivals = np.zeros(total_hours, dtype=int)
    
    total_arrived = 0    
    total_served = 0     
    total_lost = 0       
    total_revenue = 0    
    total_service_time = 0  
    waiting_times = []   
    customer_records = [] 
    
    next_arrival = generate_nhpp_arrival(t, rng, daily_lambda_non_peak, daily_lambda_max_peak)
    
    while t < SIM_END:
        min_depart_time = min(server_depart_time)
        
        if next_arrival < min_depart_time:
            t = next_arrival
            if t >= SIM_END:  
                break
            total_arrived += 1
            
            # 提前生成該顧客的所有隨機變量（保證RNG步調一致）
            revenue = rng.lognormal(LOGNORMAL_MU, LOGNORMAL_SIGMA)
            service_time = rng.exponential(1 / SERVICE_MEAN_THEORY)
            
            next_arrival = generate_nhpp_arrival(t, rng, daily_lambda_non_peak, daily_lambda_max_peak)
            
            hour_idx = int((t - SIM_START) // 60)
            if 0 <= hour_idx < total_hours:
                hourly_arrivals[hour_idx] += 1
            
            current_available_cap = active_servers * QUEUE_CAP_PER_SERVER
            total_queue_size = sum(len(q) for q in queues[:active_servers])
            if total_queue_size >= current_available_cap:
                total_lost += 1
                continue  # 即使流失，也已消耗隨機數，保證同步
            
            available_queues = [i for i in range(active_servers) if len(queues[i]) < QUEUE_CAP_PER_SERVER]
            if not available_queues and active_servers < MAX_SERVERS:
                active_servers += 1
                available_queues = [active_servers - 1]
            
            if available_queues:
                assigned_queue = min(available_queues, key=lambda x: len(queues[x]))
                queues[assigned_queue].append((t, revenue, service_time))
                
                all_queues_ge_threshold = all(len(queues[i]) >= OPEN_THRESHOLD for i in range(active_servers))
                if all_queues_ge_threshold and active_servers < MAX_SERVERS:
                    active_servers += 1
            else:
                total_lost += 1
                continue
            
            if not server_busy[assigned_queue]:
                server_busy[assigned_queue] = True
                arrival_time, rev, svc_time = queues[assigned_queue].pop(0)
                wait_time = 0.0
                waiting_times.append(wait_time)
                
                total_service_time += svc_time
                total_revenue += rev
                total_served += 1
                server_depart_time[assigned_queue] = t + svc_time
                customer_records.append((wait_time, svc_time, rev))
        else:
            server_idx = server_depart_time.index(min_depart_time)
            if min_depart_time == float('inf'):
                break
            t = min_depart_time
            
            if len(queues[server_idx]) > 0:
                arrival_time, rev, svc_time = queues[server_idx].pop(0)
                wait_time = t - arrival_time
                waiting_times.append(wait_time)
                
                total_service_time += svc_time
                total_revenue += rev
                total_served += 1
                server_depart_time[server_idx] = t + svc_time
                customer_records.append((wait_time, svc_time, rev))
            else:
                server_busy[server_idx] = False
                server_depart_time[server_idx] = float('inf')
                server_idle_start[server_idx] = t
            
            def check_idle_servers():
                nonlocal active_servers
                for s_idx in range(active_servers-1, MIN_SERVERS-1, -1):
                    if not server_busy[s_idx] and len(queues[s_idx]) == 0:
                        if server_idle_start[s_idx] == float('inf'):
                            server_idle_start[s_idx] = t
                        else:
                            idle_duration = t - server_idle_start[s_idx]
                            if idle_duration >= IDLE_CLOSE_THRESHOLD:
                                active_servers -= 1
                                server_idle_start[s_idx] = float('inf')
                                server_depart_time[s_idx] = float('inf')
                    else:
                        server_idle_start[s_idx] = float('inf')
            
            check_idle_servers()
    
    if process_remaining_queue:
        while any(server_busy) or any(len(q) > 0 for q in queues):
            valid_depart_times = [dt for dt in server_depart_time if dt != float('inf')]
            if not valid_depart_times:
                break
            
            min_depart_time = min(valid_depart_times)
            server_idx = server_depart_time.index(min_depart_time)
            t = min_depart_time
            
            if len(queues[server_idx]) > 0:
                arrival_time, rev, svc_time = queues[server_idx].pop(0)
                wait_time = t - arrival_time
                waiting_times.append(wait_time)
                
                total_service_time += svc_time
                total_revenue += rev
                total_served += 1
                server_depart_time[server_idx] = t + svc_time
                customer_records.append((wait_time, svc_time, rev))
            else:
                server_busy[server_idx] = False
                server_depart_time[server_idx] = float('inf')
                server_idle_start[server_idx] = float('inf')
    
    avg_wait_time = np.mean(waiting_times) if waiting_times else 0.0
    loss_rate = total_lost / total_arrived if total_arrived > 0 else 0.0
    avg_revenue = total_revenue / total_served if total_served > 0 else 0.0
    avg_service_time = total_service_time / total_served if total_served > 0 else 0.0
    
    return {
        "總到達顧客數": total_arrived,
        "總完成服務數": total_served,
        "總放棄顧客數": total_lost,
        "平均等待時間(分鐘)": avg_wait_time,
        "顧客流失率": loss_rate,
        "平均消費金額(元)": avg_revenue,
        "總營收(元)": total_revenue,
        "服務時間總和(分鐘)": total_service_time,
        "平均服務時間(分鐘)": avg_service_time,
        "顧客個體記錄": customer_records,
        "每小時到達人數": hourly_arrivals
    }

# ==============================================================================
# 6. 預模擬：基於個體級別數據估計控制變量最優係數
# ==============================================================================
def estimate_control_coefficients(pre_runs=PRE_RUNS):
    """預模擬：僅計算等待時間(Y)與服務時間(X)的單變量協方差與方差"""
    all_wait_times = []
    all_service_times = []
    
    print(f"\n開始預模擬（共{pre_runs}次），收集個體級別數據...")
    for run in range(pre_runs):
        res = simulate_queue(seed=run + SEED_OFFSET, use_dual=False)
        records = res["顧客個體記錄"]
        
        if len(records) > 0:
            for wait, service, _ in records:
                all_wait_times.append(wait)
                all_service_times.append(service)
    
    if not all_wait_times:
        print("警告：預模擬未收集到有效數據，使用默認係數")
        return 0.0
    
    Y = np.array(all_wait_times)
    X = np.array(all_service_times)
    
    var_X = np.var(X, ddof=1)
    if var_X == 0:
        c1 = 0.0
    else:
        cov_YX = np.cov(Y, X)[0, 1]
        c1 = -cov_YX / var_X
    
    print(f"估計的控制變量最優係數：c1(服務時間)={c1:.4f}")
    return c1

# ==============================================================================
# 7. 正式模擬：結合對偶變量+控制變量的方差縮減
# ==============================================================================
def run_formal_simulation(num_runs=NUM_RUNS, c1=0.0):
    """
    正式模擬：結合對偶變量和單一控制變量(服務時間)實現方差縮減
    """
    print(f"\n開始正式模擬（共{num_runs}次），應用方差縮減技術...")
    all_results = {
        "平均等待時間": [],
        "顧客流失率": [],
        "總營收": [],
        "平均消費金額": [],
        "總到達顧客數": [],
        "總放棄顧客數": [],
        "每小時到達人數": []
    }
    
    pre_service_mean = SERVICE_MEAN_THEORY
    
    for run in range(num_runs):
        res_original = simulate_queue(seed=run, use_dual=False)
        res_dual = simulate_queue(seed=run, use_dual=True)
        
        records_original = res_original["顧客個體記錄"]
        records_dual = res_dual["顧客個體記錄"]
        
        # 控制變量調整（僅針對等待時間，僅使用服務時間作為控制變量）
        def adjust_wait_time(records, c1, mu_s):
            if not records:
                return 0.0
            wait_times = [r[0] for r in records]
            service_times = [r[1] for r in records]
            
            # 控制變量調整公式簡化
            adjusted = np.mean(wait_times) + c1 * (np.mean(service_times) - mu_s)
            return adjusted
        
        wait_original_adjusted = adjust_wait_time(records_original, c1, pre_service_mean)
        wait_dual_adjusted = adjust_wait_time(records_dual, c1, pre_service_mean)
        final_wait_time = (wait_original_adjusted + wait_dual_adjusted) / 2
        
        all_results["平均等待時間"].append(final_wait_time)
        all_results["顧客流失率"].append((res_original["顧客流失率"] + res_dual["顧客流失率"]) / 2)
        all_results["總營收"].append((res_original["總營收(元)"] + res_dual["總營收(元)"]) / 2)
        all_results["平均消費金額"].append((res_original["平均消費金額(元)"] + res_dual["平均消費金額(元)"]) / 2)
        all_results["總到達顧客數"].append((res_original["總到達顧客數"] + res_dual["總到達顧客數"]) / 2)
        all_results["總放棄顧客數"].append((res_original["總放棄顧客數"] + res_dual["總放棄顧客數"]) / 2)
        
        hourly_original = res_original["每小時到達人數"]
        hourly_dual = res_dual["每小時到達人數"]
        hourly_avg = (hourly_original + hourly_dual) / 2
        all_results["每小時到達人數"].append(hourly_avg)
        
        if (run+1) % 20 == 0:
            print(f"已完成 {run+1}/{num_runs} 次模擬")
    
    return all_results

# ==============================================================================
# 計算95%置信區間的輔助函數
# ==============================================================================
def calculate_95_ci(data):
    data = np.array(data)
    n = len(data)
    if n < 2:
        return (np.mean(data), 0.0, 0.0)
    
    mean = np.mean(data)
    std_err = stats.sem(data)
    t_critical = stats.t.ppf(0.975, df=n-1)
    margin = t_critical * std_err
    ci_lower = mean - margin
    ci_upper = mean + margin
    
    return (mean, ci_lower, ci_upper)

# ==============================================================================
# 8. 結果分析與可視化
# ==============================================================================
def analyze_and_visualize(results):
    wait_times = results["平均等待時間"]
    loss_rates = results["顧客流失率"]
    total_revenues = results["總營收"]
    avg_consumptions = results["平均消費金額"]
    total_arrivals = results["總到達顧客數"]
    total_abandons = results["總放棄顧客數"]
    
    wait_mean, wait_ci_l, wait_ci_u = calculate_95_ci(wait_times)
    loss_mean, loss_ci_l, loss_ci_u = calculate_95_ci(loss_rates)
    revenue_mean, revenue_ci_l, revenue_ci_u = calculate_95_ci(total_revenues)
    consume_mean, consume_ci_l, consume_ci_u = calculate_95_ci(avg_consumptions)
    arrival_mean, arrival_ci_l, arrival_ci_u = calculate_95_ci(total_arrivals)
    abandon_mean, abandon_ci_l, abandon_ci_u = calculate_95_ci(total_abandons)
    
    print("\n==================== 模擬結果匯總 ====================")
    print(f"平均等待時間：{wait_mean:.4f} 分鐘 | 95%置信區間：[{wait_ci_l:.4f}, {wait_ci_u:.4f}]")
    print(f"顧客流失率：{loss_mean:.4%} | 95%置信區間：[{loss_ci_l:.4%}, {loss_ci_u:.4%}]")
    print(f"平均消費金額：{consume_mean:.2f} 元 | 95%置信區間：[{consume_ci_l:.2f}, {consume_ci_u:.2f}]")
    print(f"總營收均值：{revenue_mean:.2f} 元 | 95%置信區間：[{revenue_ci_l:.2f}, {revenue_ci_u:.2f}]")
    print(f"平均總客量：{arrival_mean:.2f} 人 | 95%置信區間：[{arrival_ci_l:.2f}, {arrival_ci_u:.2f}]")
    print(f"平均流失客量：{abandon_mean:.2f} 人 | 95%置信區間：[{abandon_ci_l:.2f}, {abandon_ci_u:.2f}]")
    print("======================================================")
    
    #中文
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.hist(wait_times, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    ax1.axvline(wait_mean, color="red", linestyle="--", linewidth=2, label=f"均值：{wait_mean:.4f} 分鐘")
    ax1.axvline(wait_ci_l, color="orange", linestyle=":", linewidth=2, label=f"95%CI下限：{wait_ci_l:.4f}")
    ax1.axvline(wait_ci_u, color="orange", linestyle=":", linewidth=2, label=f"95%CI上限：{wait_ci_u:.4f}")
    ax1.axvspan(wait_ci_l, wait_ci_u, color="orange", alpha=0.1, label="95%置信區間")
    ax1.set_title("平均等待時間分佈", fontsize=12, fontweight="bold")
    ax1.set_xlabel("等待時間（分鐘）")
    ax1.set_ylabel("次數")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.hist(total_revenues, bins=20, color="lightgreen", edgecolor="black", alpha=0.7)
    ax2.axvline(revenue_mean, color="red", linestyle="--", linewidth=2, label=f"均值：{revenue_mean:.2f} 元")
    ax2.axvline(revenue_ci_l, color="orange", linestyle=":", linewidth=2, label=f"95%CI下限：{revenue_ci_l:.2f}")
    ax2.axvline(revenue_ci_u, color="orange", linestyle=":", linewidth=2, label=f"95%CI上限：{revenue_ci_u:.2f}")
    ax2.axvspan(revenue_ci_l, revenue_ci_u, color="orange", alpha=0.1, label="95%置信區間")
    ax2.set_title("總營收分佈", fontsize=12, fontweight="bold")
    ax2.set_xlabel("營收（元）")
    ax2.set_ylabel("次數")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.scatter(wait_times, total_revenues, color="orange", alpha=0.6, s=30)
    if len(wait_times) > 1:
        z = np.polyfit(wait_times, total_revenues, 1)
        p = np.poly1d(z)
        ax3.plot(wait_times, p(wait_times), "b--", alpha=0.5, 
                 label=f"趨勢線：y={z[0]:.2f}x + {z[1]:.2f}")
    ax3.set_title("平均等待時間 vs 總營收", fontsize=12, fontweight="bold")
    ax3.set_xlabel("平均等待時間（分鐘）")
    ax3.set_ylabel("總營收（元）")
    ax3.legend(loc="best")
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    hourly_arrivals_all = np.array(results["每小時到達人數"])
    hourly_means = []
    hourly_ci_lowers = []
    hourly_ci_uppers = []
    for hour_idx in range(hourly_arrivals_all.shape[1]):
        hour_data = hourly_arrivals_all[:, hour_idx]
        h_mean, h_ci_l, h_ci_u = calculate_95_ci(hour_data)
        hourly_means.append(h_mean)
        hourly_ci_lowers.append(h_ci_l)
        hourly_ci_uppers.append(h_ci_u)
    
    start_hour = int(SIM_START / 60)
    end_hour = int(SIM_END / 60)
    hour_labels = [f"{h:02d}:00" for h in range(start_hour, end_hour)]
    x_pos = np.arange(len(hour_labels))
    
    fig4, ax4 = plt.subplots(figsize=(11, 6))
    bars = ax4.bar(
        x_pos, hourly_means, 
        width=0.6, color="cornflowerblue", 
        edgecolor="black", alpha=0.7, 
        label="平均到達人數"
    )
    ax4.errorbar(
        x_pos, hourly_means,
        yerr=[np.array(hourly_means)-np.array(hourly_ci_lowers), 
              np.array(hourly_ci_uppers)-np.array(hourly_means)],
        fmt="none", c="black", capsize=5, 
        label="95%置信區間"
    )
    ax4.set_title("每小時平均到達人流分佈", fontsize=12, fontweight="bold")
    ax4.set_xlabel("營業時間（小時）")
    ax4.set_ylabel("平均到達人數")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(hour_labels)
    ax4.legend(loc="upper right")
    ax4.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 9. 主函數：執行完整流程
# ==============================================================================
if __name__ == "__main__":
    # 步驟1：估計控制變量係數（僅返回 c1）
    c1_opt = estimate_control_coefficients(pre_runs=PRE_RUNS)
    
    # 步驟2：執行正式模擬（僅傳入 c1）
    formal_results = run_formal_simulation(num_runs=NUM_RUNS, c1=c1_opt)
    
    # 步驟3：分析結果並可視化
    analyze_and_visualize(formal_results)