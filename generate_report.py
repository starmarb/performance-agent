#!/usr/bin/env python3
"""
PERFORMANCE AGENT - Report Generator

Generates a detailed, readable report from analysis JSON files.

Usage:
    python generate_report.py analysis_results_multi.json
    python generate_report.py analysis_laneige_jp.json
"""

import json
import sys
from datetime import datetime

# Content type labels
CONTENT_LABELS = {
    "product_group": "제품 단체샷 (Product Group Shot)",
    "product_solo": "제품 단독샷 (Product Solo Shot)",
    "product_texture": "제품 질감샷 (Product Texture Shot)",
    "product_model": "제품 및 모델샷 (Product + Model Shot)",
    "model_only": "제품없는 모델샷 (Model Only Shot)",
    "product_models_group": "제품 모델 단체샷 (Multi-Model Group Shot)"
}

CONTENT_LABELS_SHORT = {
    "product_group": "제품 단체샷",
    "product_solo": "제품 단독샷",
    "product_texture": "제품 질감샷",
    "product_model": "제품 및 모델샷",
    "model_only": "제품없는 모델샷",
    "product_models_group": "제품 모델 단체샷"
}

COUNTRY_FLAGS = {
    "laneige_jp": "🇯🇵",
    "laneigethailand": "🇹🇭",
    "laneigesg": "🇸🇬"
}

COUNTRY_NAMES = {
    "laneige_jp": "Japan",
    "laneigethailand": "Thailand",
    "laneigesg": "Singapore"
}


def generate_single_account_report(data: dict) -> str:
    """Generate report for single account analysis"""
    
    account = data["account"]
    posts = data["posts"]
    insights = data["insights"]
    
    username = account["username"]
    flag = COUNTRY_FLAGS.get(username, "")
    country = COUNTRY_NAMES.get(username, "")
    
    report = []
    
    # Header
    report.append("=" * 70)
    report.append(f"PERFORMANCE AGENT ANALYSIS REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 70)
    report.append("")
    
    # Account Overview
    report.append(f"## {flag} @{username} ({country}) Performance Analysis")
    report.append("")
    report.append(f"Account: {account['name']}")
    report.append(f"Followers: {account['followers']:,}")
    report.append(f"Posts Analyzed: {insights['total_posts_analyzed']}")
    report.append("")
    
    # Key Findings
    report.append("-" * 70)
    report.append("### KEY FINDINGS")
    report.append("-" * 70)
    report.append("")
    
    top_type = insights['top_performing_type']
    bottom_type = insights['bottom_performing_type']
    preference = insights['content_preference']
    
    report.append(f"📊 Average Engagement Rate: {insights['avg_engagement_rate']}%")
    report.append("")
    report.append(f"🏆 Best Performing Content Type:")
    report.append(f"   {CONTENT_LABELS.get(top_type['type'], top_type['type'])}")
    report.append(f"   Average Engagement: {top_type['avg_engagement']}%")
    report.append("")
    report.append(f"📉 Lowest Performing Content Type:")
    report.append(f"   {CONTENT_LABELS.get(bottom_type['type'], bottom_type['type'])}")
    report.append(f"   Average Engagement: {bottom_type['avg_engagement']}%")
    report.append("")
    
    # Preference Analysis
    pref_display = "Product-Focused 📦" if preference == "product_focused" else "Human-Focused 👤"
    report.append(f"🎯 Content Preference: {pref_display}")
    report.append(f"   Preference Strength: {insights['preference_strength']:.2f}% difference")
    report.append("")
    
    # Engagement by Content Type Table
    report.append("-" * 70)
    report.append("### ENGAGEMENT BY CONTENT TYPE")
    report.append("-" * 70)
    report.append("")
    report.append(f"{'Content Type':<35} {'Engagement':>12} {'Rank':>8}")
    report.append("-" * 55)
    
    sorted_types = sorted(
        insights['engagement_by_content_type'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for rank, (ct, eng) in enumerate(sorted_types, 1):
        label = CONTENT_LABELS_SHORT.get(ct, ct)
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        report.append(f"{medal} {label:<32} {eng:>10.2f}% {rank:>6}")
    
    report.append("")
    
    # Individual Post Analysis
    report.append("-" * 70)
    report.append("### INDIVIDUAL POST ANALYSIS")
    report.append("-" * 70)
    report.append("")
    
    # Sort posts by engagement
    sorted_posts = sorted(posts, key=lambda x: x['analysis']['engagement_rate'], reverse=True)
    
    for i, post in enumerate(sorted_posts, 1):
        analysis = post['analysis']
        metrics = post['metrics']
        vision = analysis.get('vision', {})
        
        content_type = vision.get('content_type', analysis.get('content_type', 'unknown'))
        
        report.append(f"Post #{i} (Rank by Engagement)")
        report.append(f"   File: {post['local_file']}")
        report.append(f"   Content Type: {CONTENT_LABELS_SHORT.get(content_type, content_type)}")
        report.append("")
        report.append(f"   📈 Metrics:")
        report.append(f"      Likes: {metrics['likes']:,}")
        report.append(f"      Comments: {metrics['comments']:,}")
        report.append(f"      Saves: {metrics['saves']:,}")
        report.append(f"      Shares: {metrics['shares']:,}")
        report.append(f"      Reach: {metrics['reach']:,}")
        report.append(f"      Engagement Rate: {analysis['engagement_rate']}%")
        report.append(f"      Conversion Score: {analysis['conversion_score']}/10")
        report.append("")
        
        # Vision Analysis
        if vision and vision.get('human_presence') is not None:
            report.append(f"   🖼️ Visual Analysis:")
            report.append(f"      Human Presence: {'Yes' if vision.get('human_presence') else 'No'}")
            report.append(f"      Product Visibility: {vision.get('product_visibility', 'N/A')}")
            report.append(f"      Composition: {vision.get('composition_style', 'N/A')}")
            report.append(f"      Mood: {vision.get('mood', 'N/A')}")
            report.append("")
        
        # Caption Analysis
        caption = analysis.get('caption', {})
        if caption:
            report.append(f"   📝 Caption Analysis:")
            report.append(f"      Language: {caption.get('language', 'N/A').upper()}")
            report.append(f"      Hashtags: {caption.get('hashtag_count', 0)}")
            report.append(f"      Emojis: {caption.get('emoji_count', 0)}")
            report.append(f"      Has CTA: {'Yes' if caption.get('cta_present') else 'No'}")
            report.append(f"      Tone: {caption.get('tone', 'N/A')}")
            report.append("")
        
        # Comment Analysis
        comments = analysis.get('comments', {})
        if comments:
            sentiment = comments.get('sentiment_score', 0)
            sentiment_emoji = "😊" if sentiment > 0.6 else "😐" if sentiment > 0.3 else "😟"
            report.append(f"   💬 Comment Analysis:")
            report.append(f"      Total Comments: {comments.get('total_comments', 0)}")
            report.append(f"      Sentiment: {sentiment_emoji} {sentiment:.0%} positive")
            report.append(f"      Purchase Intent: {comments.get('purchase_intent_count', 0)} comments")
            report.append(f"      Questions: {comments.get('questions_count', 0)} comments")
            if comments.get('top_topics'):
                report.append(f"      Top Topics: {', '.join(comments['top_topics'][:3])}")
            report.append("")
        
        report.append("-" * 40)
        report.append("")
    
    # Recommendations
    report.append("-" * 70)
    report.append("### RECOMMENDATIONS")
    report.append("-" * 70)
    report.append("")
    report.append(f"💡 {insights['recommendation']}")
    report.append("")
    
    # Generate specific recommendations based on data
    if preference == "product_focused":
        report.append("Based on the analysis, we recommend:")
        report.append("")
        report.append("   1. Prioritize product-focused imagery")
        report.append("      - Hero shots showing product details")
        report.append("      - Texture close-ups that highlight product quality")
        report.append("      - Clean, minimal backgrounds")
        report.append("")
        report.append("   2. When using models, keep focus on the product")
        report.append("      - Product should be prominently visible")
        report.append("      - Avoid lifestyle shots where product is secondary")
        report.append("")
    else:
        report.append("Based on the analysis, we recommend:")
        report.append("")
        report.append("   1. Include human elements in content")
        report.append("      - Models demonstrating product usage")
        report.append("      - Authentic skincare routine scenes")
        report.append("      - Diverse representation resonates well")
        report.append("")
        report.append("   2. Balance product visibility with human connection")
        report.append("      - Show real results on skin")
        report.append("      - Create aspirational but relatable imagery")
        report.append("")
    
    # Highlight top performing post
    best_post = sorted_posts[0]
    report.append(f"   3. Model content after top performer:")
    report.append(f"      - {best_post['local_file']}")
    report.append(f"      - Achieved {best_post['analysis']['engagement_rate']}% engagement")
    report.append("")
    
    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    return "\n".join(report)


def generate_multi_account_report(data: dict) -> str:
    """Generate report for multi-account analysis"""
    
    accounts_data = data["accounts"]
    comparison = data["comparison"]
    
    report = []
    
    # Header
    report.append("=" * 70)
    report.append("PERFORMANCE AGENT")
    report.append("CROSS-MARKET ANALYSIS REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 70)
    report.append("")
    
    # Executive Summary
    report.append("-" * 70)
    report.append("## EXECUTIVE SUMMARY")
    report.append("-" * 70)
    report.append("")
    
    report.append(f"Markets Analyzed: {len(accounts_data)}")
    for acc_id in accounts_data:
        flag = COUNTRY_FLAGS.get(acc_id, "")
        country = COUNTRY_NAMES.get(acc_id, acc_id)
        report.append(f"   {flag} {country} (@{acc_id})")
    report.append("")
    
    # Key insight summary
    report.append("Key Discovery:")
    
    product_focused = []
    human_focused = []
    
    for acc_id, acc_data in accounts_data.items():
        pref = acc_data['insights']['content_preference']
        country = COUNTRY_NAMES.get(acc_id, acc_id)
        flag = COUNTRY_FLAGS.get(acc_id, "")
        if pref == "product_focused":
            product_focused.append(f"{flag} {country}")
        else:
            human_focused.append(f"{flag} {country}")
    
    if product_focused:
        report.append(f"   📦 Product-focused markets: {', '.join(product_focused)}")
    if human_focused:
        report.append(f"   👤 Human-focused markets: {', '.join(human_focused)}")
    report.append("")
    
    # Comparison Table
    report.append("-" * 70)
    report.append("## MARKET COMPARISON")
    report.append("-" * 70)
    report.append("")
    
    # Header row
    header = f"{'Metric':<25} "
    for acc_id in accounts_data:
        flag = COUNTRY_FLAGS.get(acc_id, "")
        header += f"{flag + ' ' + COUNTRY_NAMES.get(acc_id, acc_id):<18} "
    report.append(header)
    report.append("-" * (25 + 18 * len(accounts_data)))
    
    # Engagement row
    row = f"{'Avg Engagement':<25} "
    for acc_id, acc_data in accounts_data.items():
        eng = acc_data['insights']['avg_engagement_rate']
        row += f"{eng:<18.2f}%"
    report.append(row)
    
    # Top Content row
    row = f"{'Top Content Type':<25} "
    for acc_id, acc_data in accounts_data.items():
        top = acc_data['insights']['top_performing_type']['type']
        label = CONTENT_LABELS_SHORT.get(top, top)[:15]
        row += f"{label:<18} "
    report.append(row)
    
    # Preference row
    row = f"{'Content Preference':<25} "
    for acc_id, acc_data in accounts_data.items():
        pref = acc_data['insights']['content_preference']
        display = "Product 📦" if pref == "product_focused" else "Human 👤"
        row += f"{display:<18} "
    report.append(row)
    
    report.append("")
    
    # Detailed breakdown per market
    report.append("-" * 70)
    report.append("## DETAILED MARKET ANALYSIS")
    report.append("-" * 70)
    
    for acc_id, acc_data in accounts_data.items():
        flag = COUNTRY_FLAGS.get(acc_id, "")
        country = COUNTRY_NAMES.get(acc_id, acc_id)
        insights = acc_data['insights']
        account = acc_data['account']
        
        report.append("")
        report.append(f"### {flag} {country} (@{acc_id})")
        report.append("")
        report.append(f"Followers: {account['followers']:,}")
        report.append(f"Posts Analyzed: {insights['total_posts_analyzed']}")
        report.append(f"Average Engagement: {insights['avg_engagement_rate']}%")
        report.append("")
        
        # Content type performance
        report.append("Content Type Performance (sorted by engagement):")
        report.append("")
        
        sorted_types = sorted(
            insights['engagement_by_content_type'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for rank, (ct, eng) in enumerate(sorted_types, 1):
            label = CONTENT_LABELS_SHORT.get(ct, ct)
            bar_length = int(eng * 3)  # Scale for visual bar
            bar = "█" * bar_length
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            report.append(f"   {medal} {label:<20} {eng:>6.2f}% {bar}")
        
        report.append("")
        
        # Top performing post
        posts = acc_data['posts']
        top_post = max(posts, key=lambda x: x['analysis']['engagement_rate'])
        report.append(f"🏆 Top Performing Post:")
        report.append(f"   File: {top_post['local_file']}")
        report.append(f"   Engagement: {top_post['analysis']['engagement_rate']}%")
        report.append(f"   Likes: {top_post['metrics']['likes']:,} | Saves: {top_post['metrics']['saves']:,}")
        report.append("")
        
        # Bottom performing post
        bottom_post = min(posts, key=lambda x: x['analysis']['engagement_rate'])
        report.append(f"📉 Lowest Performing Post:")
        report.append(f"   File: {bottom_post['local_file']}")
        report.append(f"   Engagement: {bottom_post['analysis']['engagement_rate']}%")
        report.append("")
        
        report.append("-" * 40)
    
    # Cross-Market Insights
    report.append("")
    report.append("-" * 70)
    report.append("## CROSS-MARKET INSIGHTS & RECOMMENDATIONS")
    report.append("-" * 70)
    report.append("")
    
    for rec in comparison['cross_market_recommendations']:
        report.append(f"• {rec}")
    report.append("")
    
    # Strategic recommendations
    report.append("Strategic Recommendations:")
    report.append("")
    
    if product_focused and human_focused:
        report.append("   🎯 Localization Strategy Required")
        report.append("")
        report.append("   The data shows clear regional preferences that require")
        report.append("   differentiated content strategies:")
        report.append("")
        
        for country in product_focused:
            report.append(f"   For {country}:")
            report.append("      • Lead with product hero shots")
            report.append("      • Emphasize texture and formulation")
            report.append("      • Minimal human presence in imagery")
            report.append("")
        
        for country in human_focused:
            report.append(f"   For {country}:")
            report.append("      • Include models in content")
            report.append("      • Show product in use (application shots)")
            report.append("      • Diverse representation drives engagement")
            report.append("")
    
    report.append("   📊 Content Production Efficiency")
    report.append("")
    report.append("   To maximize ROI on content production:")
    report.append("   1. Create product-only base assets (works for product-focused markets)")
    report.append("   2. Create model variants for human-focused markets")
    report.append("   3. Test texture/detail shots across all markets")
    report.append("")
    
    # Conclusion
    report.append("-" * 70)
    report.append("## CONCLUSION")
    report.append("-" * 70)
    report.append("")
    report.append("This analysis reveals significant differences in content preferences")
    report.append("across markets. By tailoring content strategy to each market's")
    report.append("preferences, LANEIGE can optimize engagement and conversion rates")
    report.append("while maintaining brand consistency.")
    report.append("")
    report.append("The PERFORMANCE AGENT enables data-driven content decisions by:")
    report.append("   • Automatically classifying content types using AI vision")
    report.append("   • Analyzing engagement patterns by content category")
    report.append("   • Identifying market-specific preferences")
    report.append("   • Generating actionable recommendations")
    report.append("")
    
    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    return "\n".join(report)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <analysis_file.json>")
        print("")
        print("Examples:")
        print("  python generate_report.py analysis_results_multi.json")
        print("  python generate_report.py analysis_laneige_jp.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {input_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON file: {input_file}")
        sys.exit(1)
    
    # Detect if single or multi-account
    if "accounts" in data:
        # Multi-account analysis
        report = generate_multi_account_report(data)
        output_file = input_file.replace('.json', '_report.txt')
    else:
        # Single account analysis
        report = generate_single_account_report(data)
        output_file = input_file.replace('.json', '_report.txt')
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Also print to console
    print(report)
    print("")
    print(f"📄 Report saved to: {output_file}")


if __name__ == "__main__":
    main()
