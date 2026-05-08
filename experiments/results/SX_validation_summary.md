# Validation matrix — info-theoretic scores cross-model cross-dataset

kept_acc at α=0.30 (bootstrap mean over 300×5 splits, 95% CI in [low, high]).

Vanilla = greedy accuracy. n is per-dataset sample size.


## Dataset: math500

| model | n | vanilla | lp_mean_min | top1_margin_mean | top1_margin_min | entropy_max | entropy_mean | tempered_kl_max | tempered_kl_mean | kl_uniform_min | kl_uniform_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen25_7b | 200 | 0.745 | 0.809 | 0.831 | 0.817 | 0.812 | 0.796 | 0.820 | 0.802 | 0.812 | 0.796 |
| qwen25_math_7b | 200 | 0.800 | 0.859 | 0.890 | 0.861 | 0.859 | 0.877 | 0.871 | 0.887 | 0.859 | 0.877 |
| qwen25_32b | 200 | 0.805 | 0.870 | 0.841 | 0.869 | 0.890 | 0.859 | 0.871 | 0.835 | 0.890 | 0.859 |
| phi4 | 200 | 0.775 | 0.812 | 0.803 | 0.830 | 0.826 | 0.797 | 0.804 | 0.821 | 0.826 | 0.797 |

## Dataset: aime

| model | n | vanilla | lp_mean_min | top1_margin_mean | top1_margin_min | entropy_max | entropy_mean | tempered_kl_max | tempered_kl_mean | kl_uniform_min | kl_uniform_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen25_7b | 200 | 0.290 | 0.443 | 0.484 | 0.410 | 0.442 | 0.486 | 0.425 | 0.481 | 0.442 | 0.486 |
| qwen25_math_7b | 200 | 0.415 | 0.531 | 0.563 | 0.558 | 0.551 | 0.677 | 0.559 | 0.537 | 0.551 | 0.677 |
| qwen25_32b | 200 | 0.440 | 0.496 | 0.543 | 0.486 | 0.500 | 0.518 | 0.522 | 0.527 | 0.500 | 0.518 |
| phi4 | 200 | 0.355 | 0.425 | 0.478 | 0.367 | 0.434 | 0.500 | 0.411 | 0.500 | 0.434 | 0.500 |

## Dataset: olympiad

| model | n | vanilla | lp_mean_min | top1_margin_mean | top1_margin_min | entropy_max | entropy_mean | tempered_kl_max | tempered_kl_mean | kl_uniform_min | kl_uniform_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen25_7b | 200 | 0.430 | 0.484 | 0.573 | 0.455 | 0.485 | 0.527 | 0.502 | 0.563 | 0.485 | 0.527 |
| qwen25_math_7b | 200 | 0.480 | 0.554 | 0.606 | 0.541 | 0.553 | 0.615 | 0.560 | 0.595 | 0.553 | 0.615 |
| qwen25_32b | 200 | 0.415 | 0.506 | 0.539 | 0.489 | 0.484 | 0.555 | 0.479 | 0.547 | 0.484 | 0.555 |
| phi4 | 200 | 0.490 | 0.534 | 0.618 | 0.507 | 0.559 | 0.625 | 0.545 | 0.618 | 0.559 | 0.625 |

## Dataset: mmlu_pro

| model | n | vanilla | lp_mean_min | top1_margin_mean | top1_margin_min | entropy_max | entropy_mean | tempered_kl_max | tempered_kl_mean | kl_uniform_min | kl_uniform_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen25_7b | 200 | 0.680 | 0.721 | 0.723 | 0.714 | 0.725 | 0.713 | 0.737 | 0.717 | 0.725 | 0.713 |
| qwen25_math_7b | 200 | 0.425 | 0.451 | 0.455 | 0.467 | 0.443 | 0.467 | 0.485 | 0.454 | 0.443 | 0.467 |
| qwen25_32b | 200 | 0.845 | 0.886 | 0.861 | 0.854 | 0.887 | 0.869 | 0.876 | 0.874 | 0.887 | 0.869 |
| phi4 | 200 | 0.855 | 0.880 | 0.861 | 0.878 | 0.883 | 0.872 | 0.867 | 0.867 | 0.883 | 0.872 |

## Best score per (model, dataset)

| model | dataset | n | vanilla | best score | best kept@0.3 | lp_min kept@0.3 | Δ vs lp_min |
|---|---|---|---|---|---|---|---|
| qwen25_7b | math500 | 200 | 0.745 | top1_margin_mean | 0.831 | 0.809 | +0.022 |
| qwen25_7b | aime | 200 | 0.290 | entropy_mean | 0.486 | 0.443 | +0.043 |
| qwen25_7b | olympiad | 200 | 0.430 | top1_margin_mean | 0.573 | 0.484 | +0.090 |
| qwen25_7b | mmlu_pro | 200 | 0.680 | tempered_kl_max | 0.737 | 0.721 | +0.016 |
| qwen25_math_7b | math500 | 200 | 0.800 | top1_margin_mean | 0.890 | 0.859 | +0.031 |
| qwen25_math_7b | aime | 200 | 0.415 | entropy_mean | 0.677 | 0.531 | +0.146 |
| qwen25_math_7b | olympiad | 200 | 0.480 | entropy_mean | 0.615 | 0.554 | +0.061 |
| qwen25_math_7b | mmlu_pro | 200 | 0.425 | tempered_kl_max | 0.485 | 0.451 | +0.034 |
| qwen25_32b | math500 | 200 | 0.805 | entropy_max | 0.890 | 0.870 | +0.020 |
| qwen25_32b | aime | 200 | 0.440 | top1_margin_mean | 0.543 | 0.496 | +0.047 |
| qwen25_32b | olympiad | 200 | 0.415 | entropy_mean | 0.555 | 0.506 | +0.049 |
| qwen25_32b | mmlu_pro | 200 | 0.845 | entropy_max | 0.887 | 0.886 | +0.000 |
| phi4 | math500 | 200 | 0.775 | top1_margin_min | 0.830 | 0.812 | +0.018 |
| phi4 | aime | 200 | 0.355 | tempered_kl_mean | 0.500 | 0.425 | +0.076 |
| phi4 | olympiad | 200 | 0.490 | entropy_mean | 0.625 | 0.534 | +0.090 |
| phi4 | mmlu_pro | 200 | 0.855 | entropy_max | 0.883 | 0.880 | +0.003 |

## Score win-rate (counts of being top-1 / top-3)


Total (model, dataset) cells: 16

| Score | Top-1 wins | Top-3 wins |
|---|---|---|
| lp_mean_min | 0 | 2 |
| top1_margin_mean | 4 | 7 |
| top1_margin_min | 1 | 2 |
| entropy_max | 3 | 5 |
| entropy_mean | 5 | 9 |
| tempered_kl_max | 2 | 5 |
| tempered_kl_mean | 1 | 6 |
| kl_uniform_min | 0 | 5 |
| kl_uniform_mean | 0 | 7 |