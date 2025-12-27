#!/usr/bin/env python3
"""
Simple runner script for Performance Agent
"""

from agent import create_agent
import json

def main():
    print("=" * 60)
    print("PERFORMANCE AGENT")
    print("Instagram Content Performance Analysis")
    print("=" * 60)
    print()
    
    # Create agent in demo mode
    agent = create_agent("demo", data_dir="data/raw", image_dir="images")
    
    # Accounts to analyze
    accounts = ["laneige_jp", "laneigethailand", "laneigesg"]
    
    print(f"Analyzing {len(accounts)} accounts...")
    print()
    
    # Run analysis
    results = agent.analyze_multiple_accounts(accounts)
    
    # Save results
    output_file = "analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Full results saved to: {output_file}")
    print()
    
    # Print summary
    print("=" * 60)
    print("ACCOUNT SUMMARIES")
    print("=" * 60)
    
    for account_id, data in results["accounts"].items():
        insights = data["insights"]
        print()
        print(f"📊 @{account_id}")
        print(f"   Posts Analyzed: {insights['total_posts_analyzed']}")
        print(f"   Avg Engagement Rate: {insights['avg_engagement_rate']}%")
        print(f"   Top Content Type: {insights['top_performing_type']['label']}")
        print(f"   Content Preference: {insights['content_preference']}")
        print()
        print(f"   💡 Recommendation:")
        print(f"      {insights['recommendation']}")
    
    print()
    print("=" * 60)
    print("CROSS-MARKET INSIGHTS")
    print("=" * 60)
    print()
    
    for rec in results["comparison"]["cross_market_recommendations"]:
        print(f"   • {rec}")
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
