#!/usr/bin/env python3
"""
PERFORMANCE AGENT - Production Mode

Usage:
    # Analyze with options
    python run_production.py --key YOUR_API_KEY --accounts laneige_jp
    python run_production.py --key YOUR_API_KEY --accounts all
    
    # Interactive mode
    python run_production.py --key YOUR_API_KEY
"""

import sys
import os
import json
import argparse
from datetime import datetime

# Available accounts
AVAILABLE_ACCOUNTS = {
    "1": "laneige_jp",
    "2": "laneigethailand", 
    "3": "laneigesg"
}

COUNTRY_NAMES = {
    "laneige_jp": "Japan",
    "laneigethailand": "Thailand",
    "laneigesg": "Singapore"
}


def get_accounts_interactive():
    """Prompt user to select accounts"""
    print("\n📋 Available accounts:")
    print("   1. @laneige_jp (Japan)")
    print("   2. @laneigethailand (Thailand)")
    print("   3. @laneigesg (Singapore)")
    print("   4. All accounts")
    print()
    
    choice = input("Select accounts (e.g., '1' or '1,2' or '4' for all): ").strip()
    
    if choice == "4" or choice.lower() == "all":
        return list(AVAILABLE_ACCOUNTS.values())
    
    selected = []
    for c in choice.split(","):
        c = c.strip()
        if c in AVAILABLE_ACCOUNTS:
            selected.append(AVAILABLE_ACCOUNTS[c])
        elif c in AVAILABLE_ACCOUNTS.values():
            selected.append(c)
    
    if not selected:
        print("Invalid selection. Using laneige_jp as default.")
        return ["laneige_jp"]
    
    return selected


def parse_accounts(accounts_str):
    """Parse account string from command line"""
    if accounts_str.lower() == "all":
        return list(AVAILABLE_ACCOUNTS.values())
    
    accounts = []
    for acc in accounts_str.split(","):
        acc = acc.strip()
        if acc in AVAILABLE_ACCOUNTS.values():
            accounts.append(acc)
        elif acc in AVAILABLE_ACCOUNTS:
            accounts.append(AVAILABLE_ACCOUNTS[acc])
    
    return accounts if accounts else ["laneige_jp"]


def estimate_cost(num_accounts, posts_per_account=6):
    """Estimate API cost"""
    total_posts = num_accounts * posts_per_account
    cost = total_posts * 0.008
    return round(cost, 2)


def generate_ai_report(results: dict, api_key: str, is_multi: bool) -> str:
    """Use Claude to generate an intelligent report from the analysis data"""
    
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    
    # Prepare summary data for Claude
    if is_multi:
        summary = {
            "type": "multi_market_analysis",
            "markets": []
        }
        for acc_id, acc_data in results["accounts"].items():
            market_summary = {
                "account": acc_id,
                "country": COUNTRY_NAMES.get(acc_id, acc_id),
                "followers": acc_data["account"]["followers"],
                "insights": acc_data["insights"],
                "top_post": max(acc_data["posts"], key=lambda x: x["analysis"]["engagement_rate"]),
                "bottom_post": min(acc_data["posts"], key=lambda x: x["analysis"]["engagement_rate"])
            }
            summary["markets"].append(market_summary)
        summary["cross_market"] = results["comparison"]
    else:
        summary = {
            "type": "single_market_analysis",
            "account": results["account"],
            "insights": results["insights"],
            "posts": results["posts"]
        }
    
    prompt = f"""You are a social media analytics expert writing a report for LANEIGE's marketing team.
Based on the following Instagram performance data, write a comprehensive analysis report in Korean.
The report should be professional but readable, with clear insights and actionable recommendations.

DATA:
{json.dumps(summary, ensure_ascii=False, indent=2)}

REPORT REQUIREMENTS:
1. Write in Korean (this is for Amorepacific, a Korean makeup brand company)
2. Start with an executive summary (핵심 요약)
3. Include specific numbers and percentages to back up insights
4. Explain WHY certain content types perform better (not just what)
5. Provide actionable recommendations for each market
6. Use a professional but engaging tone
7. Format with clear sections and headers
8. Do not include emojis. Ensure professionalism. 
9. End with strategic next steps

IMPORTANT CONTEXT:
- This analysis compares content types: 제품 단독샷, 제품 단체샷, 제품 질감샷, 제품 및 모델샷, 제품없는 모델샷, 제품 모델 단체샷
- "Product-focused" means audiences engage more with product-only imagery
- "Human-focused" means audiences engage more when models are present
- The goal is to help LANEIGE optimize content strategy per market

Write the full report now:"""

    print("\n Writing the report...")
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text


def print_quick_summary(results: dict, accounts: list):
    """Print quick summary to terminal"""
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    if len(accounts) == 1:
        insights = results["insights"]
        print(f"\n📊 @{accounts[0]}")
        print(f"   Posts Analyzed: {insights['total_posts_analyzed']}")
        print(f"   Avg Engagement: {insights['avg_engagement_rate']}%")
        print(f"   Top Content: {insights['top_performing_type']['label']}")
        print(f"   Preference: {insights['content_preference']}")
        print(f"\n   💡 {insights['recommendation']}")
    else:
        for account_id, data in results["accounts"].items():
            insights = data["insights"]
            print(f"\n📊 @{account_id}")
            print(f"   Avg Engagement: {insights['avg_engagement_rate']}%")
            print(f"   Top Content: {insights['top_performing_type']['label']}")
            print(f"   Preference: {insights['content_preference']}")
        
        print()
        print("-" * 60)
        print("CROSS-MARKET INSIGHTS")
        print("-" * 60)
        for rec in results["comparison"]["cross_market_recommendations"]:
            print(f"   • {rec}")


def main():
    parser = argparse.ArgumentParser(description="Performance Agent - Instagram Analysis")
    parser.add_argument("--key", "-k", help="Anthropic API key")
    parser.add_argument("--accounts", "-a", help="Accounts to analyze (comma-separated or 'all')")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation")
    args = parser.parse_args()
    
    # Get API key
    api_key = args.key or os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("Error: No API key provided")
        print()
        print("Usage:")
        print("  python run_production.py --key YOUR_API_KEY --accounts laneige_jp")
        print()
        print("Or set environment variable:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    
    # Check anthropic installed
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed")
        print("Run: pip install anthropic")
        sys.exit(1)
    
    # Get accounts to analyze
    if args.accounts:
        accounts = parse_accounts(args.accounts)
    else:
        accounts = get_accounts_interactive()
    
    # Show plan and confirm
    print()
    print("=" * 60)
    print("PERFORMANCE AGENT - PRODUCTION MODE")
    print("=" * 60)
    print()
    print(f"📊 Accounts to analyze: {len(accounts)}")
    for acc in accounts:
        print(f"   • @{acc}")
    print()
    estimated_cost = estimate_cost(len(accounts))
    report_cost = 0.02  # Estimate for report generation
    print(f"Estimated cost: ~${estimated_cost} (analysis) + ~${report_cost} (report) = ~${estimated_cost + report_cost}")
    print()
    
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        sys.exit(0)
    
    print()
    print("=" * 60)
    print("RUNNING ANALYSIS")
    print("=" * 60)
    
    from agent import create_agent
    
    # Create agent
    agent = create_agent(
        "production",
        data_dir="data/raw",
        image_dir="images",
        claude_api_key=api_key
    )
    
    # Analyze
    is_multi = len(accounts) > 1
    
    if is_multi:
        print(f"\n Analyzing {len(accounts)} accounts...")
        results = agent.analyze_multiple_accounts(accounts)
        output_file = "analysis_results_multi.json"
    else:
        account_id = accounts[0]
        print(f"\n Analyzing @{account_id}...")
        results = agent.analyze_account(account_id)
        output_file = f"analysis_{account_id}.json"
    
    # Save JSON results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n Results saved to: {output_file}")
    
    # Print quick summary
    print_quick_summary(results, accounts)
    
    # Ask about report generation (in interactive mode)
    if not args.no_report:
        print()
        print("=" * 60)
        generate_report = input("Generate AI-written report? (y/n): ").strip().lower()
        
        if generate_report == "y":
            report = generate_ai_report(results, api_key, is_multi)
            
            # Save report
            report_file = output_file.replace('.json', '_report.md')
            
            # Add header
            full_report = f"""# PERFORMANCE AGENT Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Data source: {output_file}

---

{report}
"""
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(full_report)
            
            print()
            print("=" * 60)
            print("AI-GENERATED REPORT")
            print("=" * 60)
            print()
            print(report)
            print()
            print("=" * 60)
            print(f"Report saved to: {report_file}")
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
