#!/usr/bin/env python3
"""Compare SEC EDGAR vs Polygon REST data samples."""

import json
import os
import pandas as pd

print("=" * 80)
print("SEC EDGAR vs Polygon REST Data Comparison")
print("=" * 80)

# -----------------------------------------------------------------------------
# REST/Polygon Sample
# -----------------------------------------------------------------------------
print("\n" + "=" * 40)
print("POLYGON REST API (rest_data/)")
print("=" * 40)

df = pd.read_csv('datasets/MACRO/rest_data/income_statements/income_statements.csv', 
                 nrows=3, on_bad_lines='skip')

print("\nFormat: Pre-cleaned CSV, one row per filing")
print(f"Columns: {len(df.columns)}")
print("\nSample (JNJ Q1 2010):")
print(f"  Ticker: {df.iloc[0]['tickers']}")
print(f"  Filing Date: {df.iloc[0]['filing_date']}")
print(f"  Period End: {df.iloc[0]['period_end']}")
print(f"  Revenue: ${df.iloc[0]['revenue']:,.0f}")
print(f"  Gross Profit: ${df.iloc[0]['gross_profit']:,.0f}")
print(f"  Operating Income: ${df.iloc[0]['operating_income']:,.0f}")
print(f"  EPS: ${df.iloc[0]['basic_earnings_per_share']:.2f}")

print("\nAvailable fields:")
print(list(df.columns))

# -----------------------------------------------------------------------------
# SEC EDGAR Sample
# -----------------------------------------------------------------------------
print("\n" + "=" * 40)
print("SEC EDGAR (sec_data/companyfacts/)")
print("=" * 40)

sec_path = 'datasets/MACRO/sec_data/companyfacts'

# Find JNJ for comparison
for f in os.listdir(sec_path):
    with open(os.path.join(sec_path, f), 'r') as file:
        data = json.load(file)
        if 'JOHNSON' in data.get('entityName', '').upper():
            print(f"\nFormat: Nested JSON, one file per company")
            print(f"File: {f}")
            print(f"Entity: {data['entityName']}")
            print(f"CIK: {data['cik']}")
            
            facts = data.get('facts', {}).get('us-gaap', {})
            print(f"Total metrics: {len(facts)}")
            
            # Show revenue data structure
            for metric in ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet']:
                if metric in facts:
                    print(f"\nMetric: {metric}")
                    print(f"  Description: {facts[metric].get('description', 'N/A')[:60]}...")
                    units = facts[metric].get('units', {})
                    if 'USD' in units:
                        recent = units['USD'][-3:]
                        print(f"  Recent values:")
                        for val in recent:
                            print(f"    {val.get('end', 'N/A')}: ${val.get('val', 0):,.0f} ({val.get('form', 'N/A')})")
                    break
            
            # Show sample metrics
            print("\nSample available metrics:")
            metrics = list(facts.keys())
            revenue_metrics = [m for m in metrics if 'Revenue' in m or 'Sales' in m][:5]
            income_metrics = [m for m in metrics if 'Income' in m or 'Earnings' in m][:5]
            print(f"  Revenue-related: {revenue_metrics}")
            print(f"  Income-related: {income_metrics}")
            break

# -----------------------------------------------------------------------------
# Key Differences
# -----------------------------------------------------------------------------
print("\n" + "=" * 40)
print("KEY DIFFERENCES")
print("=" * 40)
print("""
| Aspect | SEC EDGAR | Polygon REST |
|--------|-----------|--------------|
| Format | Nested JSON | Flat CSV |
| Structure | 1 file per company | 1 row per filing |
| Metric names | US-GAAP taxonomy | Standardized |
| Parsing | Complex (varies by company) | Simple pd.read_csv() |
| Coverage | All filers (19K+) | Major tickers (3.4K) |
| File count | 970K files | 4 CSVs |
| Processing time | Hours | Seconds |
""")
