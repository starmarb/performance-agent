#!/usr/bin/env python3
"""
PERFORMANCE AGENT - Production Mode

Usage:
    # Analyze single account
    python run_production.py --key YOUR_API_KEY --accounts laneige_jp
    
    # Analyze multiple accounts
    python run_production.py --key YOUR_API_KEY --accounts laneige_jp,laneigethailand,laneigesg
    
    # Analyze all accounts
    python run_production.py --key YOUR_API_KEY --accounts all
    
    # Interactive mode (prompts for selection)
    python run_production.py --key YOUR_API_KEY
"""

import sys
import os
import json
import argparse

# Available accounts
AVAILABLE_ACCOUNTS = {
    "1": "laneige_jp",
    "2": "laneigethailand", 
    "3": "laneigesg"
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
        print("❌ Invalid selection. Using laneige_jp as default.")
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
    # ~3 API calls per post (vision, caption, comments)
    # Vision: ~$0.005 per image
    # Text: ~$0.001 per call
    total_posts = num_accounts * posts_per_account
    cost = total_posts * 0.008  # ~$0.008 per post
    return round(cost, 2)


def main():
    parser = argparse.ArgumentParser(description="Performance Agent - Instagram Analysis")
    parser.add_argument("--key", "-k", help="Anthropic API key")
    parser.add_argument("--accounts", "-a", help="Accounts to analyze (comma-separated or 'all')")
    args = parser.parse_args()
    
    # Get API key
    api_key = args.key or os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("❌ Error: No API key provided")
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
        print("❌ Error: anthropic package not installed")
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
    print(f"💰 Estimated cost: ~${estimated_cost}")
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
    if len(accounts) == 1:
        # Single account
        account_id = accounts[0]
        print(f"\n🔍 Analyzing @{account_id}...")
        results = agent.analyze_account(account_id)
        output_file = f"analysis_{account_id}.json"
    else:
        # Multiple accounts
        print(f"\n🔍 Analyzing {len(accounts)} accounts...")
        results = agent.analyze_multiple_accounts(accounts)
        output_file = "analysis_results_multi.json"
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print summary
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    if len(accounts) == 1:
        # Single account summary
        insights = results["insights"]
        print(f"\n📊 @{accounts[0]}")
        print(f"   Posts Analyzed: {insights['total_posts_analyzed']}")
        print(f"   Avg Engagement: {insights['avg_engagement_rate']}%")
        print(f"   Top Content: {insights['top_performing_type']['label']}")
        print(f"   Preference: {insights['content_preference']}")
        print(f"\n   💡 {insights['recommendation']}")
        
        # Show content type breakdown
        print(f"\n   📈 Engagement by Content Type:")
        for ct, eng in sorted(insights['engagement_by_content_type'].items(), 
                             key=lambda x: x[1], reverse=True):
            label = {
                "product_group": "제품 단체샷",
                "product_solo": "제품 단독샷", 
                "product_texture": "제품 질감샷",
                "product_model": "제품 및 모델샷",
                "model_only": "제품없는 모델샷",
                "product_models_group": "제품 모델 단체샷"
            }.get(ct, ct)
            print(f"      {label}: {eng}%")
    
    else:
        # Multi-account summary
        for account_id, data in results["accounts"].items():
            insights = data["insights"]
            print(f"\n📊 @{account_id}")
            print(f"   Avg Engagement: {insights['avg_engagement_rate']}%")
            print(f"   Top Content: {insights['top_performing_type']['label']}")
            print(f"   Preference: {insights['content_preference']}")
        
        # Cross-market insights
        print()
        print("=" * 60)
        print("CROSS-MARKET INSIGHTS")
        print("=" * 60)
        for rec in results["comparison"]["cross_market_recommendations"]:
            print(f"   • {rec}")
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
