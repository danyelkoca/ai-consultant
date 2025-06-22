import csv
import random
from datetime import datetime, timedelta

# Parameters
target_file = "data/sales.csv"
num_stores = 10000
months = ["2024-04", "2025-05"]  # Only April 2024 and May 2025
regions = ["East", "West", "North", "South"]
income_bands = ["Low", "Mid", "High"]

header = [
    "branch_id",
    "month",
    "region",
    "sales_jpy",
    "price_change_flag",
    "income_band",
    "avg_temp_c",
    "foot_traffic",
    "manager_tenure_months",
]


def generate_row(store_idx, month_idx, base_sales, price_change_flag, income_band):
    branch_id = f"BR_{store_idx:04d}"
    month = months[month_idx]
    region = random.choice(regions)

    # Generate completely random base sales for each transaction
    base_sales = random.randint(4500000, 5200000)

    # Only apply the sales drop for low income with price change in April 2025
    if month == "2025-05" and price_change_flag == 1 and income_band == "Low":
        sales_jpy = int(base_sales * random.uniform(0.7, 0.85))  # YoY drop
    else:
        # Completely random sales with no correlation to any factors
        sales_jpy = random.randint(4500000, 5200000)

    # Generate random values for other fields with no correlations
    avg_temp_c = round(random.uniform(10, 30), 1)
    foot_traffic = random.randint(9000, 18000)
    manager_tenure_months = random.randint(6, 120)

    return [
        branch_id,
        month,
        region,
        sales_jpy,
        price_change_flag,
        income_band,
        avg_temp_c,
        foot_traffic,
        manager_tenure_months,
    ]


def main():
    with open(target_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for store_idx in range(num_stores):
            # Assign completely random store-level attributes
            # Equal probability for price change flag (to make the pattern more detectable)
            price_change_flag = random.choice([0, 1])
            # Equal probability for income bands (to make the pattern more detectable)
            income_band = random.choice(income_bands)
            # Random base sales for each store (though this is now overridden in generate_row)
            base_sales = random.randint(4500000, 5200000)

            for month_idx in range(len(months)):
                row = generate_row(
                    store_idx, month_idx, base_sales, price_change_flag, income_band
                )
                writer.writerow(row)


if __name__ == "__main__":
    main()
