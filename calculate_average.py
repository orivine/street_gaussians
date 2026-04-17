import re

def extract_metric_values(log_text, metric_name):
    """
    从日志文本中提取指定指标的所有数值。
    支持格式例如：
    SSIM :    0.9587309
    PSNR :   35.5277634
    LPIPS:    0.0590904
    PSNR*:   32.6291161
    """
    pattern = rf"{metric_name}\s*:\s*([0-9]*\.?[0-9]+)"
    matches = re.findall(pattern, log_text, flags=re.IGNORECASE)
    return [float(x) for x in matches]

def compute_average(values):
    if not values:
        return None
    return sum(values) / len(values)

def main():
    log_file = "log/run_metric_complete_speed_1_exp.log"   # 改成 .log 文件路径

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        log_text = f.read()

    ssim_values = extract_metric_values(log_text, "SSIM")
    psnr_values = extract_metric_values(log_text, "PSNR")
    lpips_values = extract_metric_values(log_text, "LPIPS")
    psnr_star_values = extract_metric_values(log_text, "PSNR\*")

    avg_ssim = compute_average(ssim_values)
    avg_psnr = compute_average(psnr_values)
    avg_lpips = compute_average(lpips_values)
    avg_psnr_star = compute_average(psnr_star_values)

    print("提取结果如下：")
    print(f"SSIM 个数 : {len(ssim_values)}")
    print(f"PSNR 个数 : {len(psnr_values)}")
    print(f"LPIPS 个数: {len(lpips_values)}")
    print(f"PSNR* 个数: {len(psnr_star_values)}")
    print()

    if avg_ssim is not None:
        print(f"Average SSIM : {avg_ssim:.7f}")
    else:
        print("未找到 SSIM 数据")

    if avg_psnr is not None:
        print(f"Average PSNR : {avg_psnr:.7f}")
    else:
        print("未找到 PSNR 数据")

    if avg_lpips is not None:
        print(f"Average LPIPS: {avg_lpips:.7f}")
    else:
        print("未找到 LPIPS 数据")

    if avg_psnr_star is not None:
        print(f"Average PSNR*: {avg_psnr_star:.7f}")
    else:        
        print("未找到 PSNR* 数据")

if __name__ == "__main__":
    main()