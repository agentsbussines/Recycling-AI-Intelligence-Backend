import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pymongo import MongoClient
import hashlib
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    MONGODB_URI = os.getenv("MONGODB_URI")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    MARKETING_DB = "MarketingAgentDB"
    INVENTORY_DB = "InventoryAgentDB"
    FINANCE_DB = "FinanceAgentDB"

# =============================================================================
# ADVANCED DATABASE MANAGER
# =============================================================================

class AdvancedDBManager:
    """Intelligent database manager with deep analytics capabilities"""
    
    def __init__(self):
        self.client = MongoClient(Config.MONGODB_URI)
        self.marketing_db = self.client[Config.MARKETING_DB]
        self.inventory_db = self.client[Config.INVENTORY_DB]
        self.finance_db = self.client[Config.FINANCE_DB]
    
    def save_facebook_ads(self, file_content: bytes, filename: str) -> Tuple[bool, str]:
        """Save and deeply analyze Facebook Ads data - supports CSV and Excel"""
        try:
            file_hash = hashlib.md5(file_content).hexdigest()

            if self.marketing_db["facebook_ads"].find_one({'file_hash': file_hash}):
                return False, "⚠️ This file was already uploaded!"

            # Try to read as Excel first, with no header to detect it dynamically
            try:
                excel_data = pd.read_excel(BytesIO(file_content), sheet_name=None, header=None)
                df = list(excel_data.values())[0]  # Get first sheet
            except:
                # Fallback to CSV if not Excel
                try:
                    df = pd.read_csv(BytesIO(file_content), sep='\t', encoding='utf-8', header=None)
                except:
                    df = pd.read_csv(BytesIO(file_content), encoding='utf-8', header=None)

            # Detect the header row by looking for a key column name
            header_row = None
            for i in range(10):  # Check first 10 rows
                if 'Nombre de la campaña' in df.iloc[i].astype(str).values:
                    header_row = i
                    break
                
            if header_row is None:
                return False, "❌ Error: Could not find expected header in the file."

            # Set the detected row as header
            df.columns = df.iloc[header_row]
            df = df.iloc[header_row + 1:].reset_index(drop=True)

            # Clean up: Strip column names, drop fully empty columns, replace empty strings with NaN
            df.columns = df.columns.astype(str).str.strip()
            df = df.loc[:, df.columns != 'nan']  # Drop columns with 'nan' as name
            df = df.replace('', np.nan)

            # Map Spanish column names to English
            column_mapping = {
                'Nombre de la campaña': 'campaign_name',
                'Nombre del conjunto de anuncios': 'adset_name',
                'Estado de la entrega': 'delivery_status',
                'Nivel de la entrega': 'delivery_level',
                'Alcance': 'reach',
                'Impresiones': 'impressions',
                'Frecuencia': 'frequency',
                'Configuración de atribución': 'attribution_setting',
                'Tipo de resultado': 'result_type',
                'Resultados': 'results',
                'Importe gastado (MXN)': 'spend_mxn',
                'Coste por resultado': 'cost_per_result',
                'Inicio': 'start_date',
                'Fin': 'end_date',
                'Inicio del informe': 'report_start',
                'Fin del informe': 'report_end'
            }

            # Rename columns
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df.rename(columns={old_name: new_name}, inplace=True)

            # Convert numeric columns
            numeric_cols = ['reach', 'impressions', 'frequency', 'results', 'spend_mxn', 'cost_per_result']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Filter to campaign-level data (not adset level)
            if 'delivery_level' in df.columns:
                campaign_df = df[df['delivery_level'] == 'campaign'].copy()
            else:
                campaign_df = df.copy()

            # Remove summary rows and clean data
            campaign_df = campaign_df[campaign_df['campaign_name'].notna()]
            campaign_df = campaign_df[~campaign_df['campaign_name'].str.contains('All|Total', case=False, na=False)]

            # Store processed data
            records = campaign_df.to_dict('records')

            all_sheets = {
                'campaigns': {
                    'data': records,
                    'columns': list(campaign_df.columns),
                    'rows': len(records)
                }
            }

            # Deep analysis
            analysis = self._deep_analyze_ads(all_sheets)

            self.marketing_db["facebook_ads"].insert_one({
                'file_hash': file_hash,
                'filename': filename,
                'uploaded_at': datetime.now().isoformat(),
                'sheets': all_sheets,
                'analysis': analysis,
                'total_rows': len(records)
            })

            return True, f"✅ Success! Analyzed {len(records)} campaigns with {campaign_df['impressions'].sum():,.0f} total impressions"

        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    def _deep_analyze_ads(self, sheets: Dict) -> Dict:
        """Deep AI-powered analysis of Facebook Ads data using LangChain"""
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=Config.OPENAI_API_KEY)
            
            # Get campaign data
            campaigns_sheet = sheets.get('campaigns', {})
            campaigns_data = campaigns_sheet.get('data', [])
            
            if not campaigns_data:
                return {
                    "error": "No campaign data found",
                    "totals": {"total_spend": 0, "total_impressions": 0, "total_clicks": 0},
                    "campaigns": [],
                    "key_findings": ["No data available"],
                    "recommendations": []
                }
            
            # Calculate totals and prepare campaign summaries
            total_spend = sum(c.get('spend_mxn', 0) for c in campaigns_data)
            total_impressions = sum(c.get('impressions', 0) for c in campaigns_data)
            total_reach = sum(c.get('reach', 0) for c in campaigns_data)
            total_results = sum(c.get('results', 0) for c in campaigns_data)
            
            # Calculate CTR and CPC
            avg_ctr = (total_results / total_impressions * 100) if total_impressions > 0 else 0
            avg_cpc = (total_spend / total_results) if total_results > 0 else 0
            
            # Prepare campaign summaries
            campaign_summaries = []
            for camp in campaigns_data:
                name = camp.get('campaign_name', 'Unknown')
                spend = camp.get('spend_mxn', 0)
                impressions = camp.get('impressions', 0)
                reach = camp.get('reach', 0)
                results = camp.get('results', 0)
                cpr = camp.get('cost_per_result', 0)
                status = camp.get('delivery_status', 'unknown')
                
                # Calculate engagement rate
                engagement_rate = (results / impressions * 100) if impressions > 0 else 0
                
                campaign_summaries.append({
                    'name': name[:80],
                    'spend': float(spend),
                    'impressions': int(impressions),
                    'reach': int(reach),
                    'clicks': int(results),
                    'ctr': round(engagement_rate, 2),
                    'cpc': round(cpr, 2),
                    'status': status,
                    'conversions': int(results),
                    'cost_per_conversion': round(cpr, 2),
                    'performance_score': round((results / max(spend, 1)) * 100, 2)
                })
            
            # Sort campaigns by performance
            campaign_summaries.sort(key=lambda x: x['performance_score'], reverse=True)
            
            # Identify best and worst
            best_campaign = campaign_summaries[0] if campaign_summaries else None
            worst_campaign = campaign_summaries[-1] if len(campaign_summaries) > 1 else None
            
            # Prepare data for AI analysis
            data_summary = {
                'total_campaigns': len(campaigns_data),
                'total_spend_mxn': total_spend,
                'total_impressions': total_impressions,
                'total_reach': total_reach,
                'total_results': total_results,
                'avg_engagement_rate': avg_ctr,
                'top_3_campaigns': campaign_summaries[:3],
                'bottom_3_campaigns': campaign_summaries[-3:] if len(campaign_summaries) > 3 else []
            }
            
            # Define LangChain prompt template
            analyze_prompt = PromptTemplate(
                input_variables=["data_summary"],
                template="""Analyze this Facebook Ads performance data for a metal recycling business in Mexico.

DATA SUMMARY:
{data_summary}

TASK: Provide strategic insights in JSON format:
{{
    "trends": {{
        "engagement_trend": "increasing/stable/decreasing - brief explanation",
        "spend_efficiency": "improving/stable/declining - brief explanation",
        "audience_response": "brief analysis of how audience is responding"
    }},
    "key_findings": [
        "Finding 1 with specific metric",
        "Finding 2 with specific metric",
        "Finding 3 with specific metric"
    ],
    "recommendations": [
        "Specific actionable recommendation 1",
        "Specific actionable recommendation 2",
        "Specific actionable recommendation 3"
    ],
    "best_campaign_reason": "Why this campaign is performing best",
    "worst_campaign_reason": "Why this campaign needs improvement"
}}

Focus on actionable insights. Use actual numbers from the data. Return ONLY valid JSON."""
            )
            
            # Create LangChain chain
            analyze_chain = LLMChain(llm=llm, prompt=analyze_prompt)
            
            # Invoke chain
            response = analyze_chain.invoke({"data_summary": json.dumps(data_summary, indent=2)})
            content = response['text'].strip()
            
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            ai_insights = json.loads(content)
            
            # Combine everything
            return {
                "campaigns": campaign_summaries,
                "totals": {
                    "total_spend": round(total_spend, 2),
                    "total_impressions": int(total_impressions),
                    "total_clicks": int(total_results),
                    "total_conversions": int(total_results),
                    "avg_ctr": round(avg_ctr, 2),
                    "avg_cpc": round(avg_cpc, 2),
                    "avg_conversion_rate": round(avg_ctr, 2)
                },
                "best_campaign": {
                    "name": best_campaign['name'] if best_campaign else "N/A",
                    "reason": ai_insights.get('best_campaign_reason', 'Highest performance score')
                },
                "worst_campaign": {
                    "name": worst_campaign['name'] if worst_campaign else "N/A",
                    "reason": ai_insights.get('worst_campaign_reason', 'Lowest performance score')
                },
                "trends": ai_insights.get('trends', {}),
                "key_findings": ai_insights.get('key_findings', []),
                "recommendations": ai_insights.get('recommendations', [])
            }
            
        except Exception as e:
            # Fallback: return basic analysis without AI
            campaigns_sheet = sheets.get('campaigns', {})
            campaigns_data = campaigns_sheet.get('data', [])
            
            total_spend = sum(c.get('spend_mxn', 0) for c in campaigns_data)
            total_impressions = sum(c.get('impressions', 0) for c in campaigns_data)
            total_results = sum(c.get('results', 0) for c in campaigns_data)
            
            return {
                "error": f"AI analysis failed: {str(e)}",
                "totals": {
                    "total_spend": round(total_spend, 2),
                    "total_impressions": int(total_impressions),
                    "total_clicks": int(total_results),
                    "total_conversions": int(total_results),
                    "avg_ctr": round((total_results / max(total_impressions, 1)) * 100, 2),
                    "avg_cpc": round(total_spend / max(total_results, 1), 2),
                    "avg_conversion_rate": 0
                },
                "campaigns": [],
                "key_findings": ["Data saved successfully - analysis in progress"],
                "recommendations": ["Upload complete - manual review recommended"]
            }
    
    def get_facebook_ads(self):
        """Get Facebook Ads data"""
        return list(self.marketing_db["facebook_ads"].find({}, {'_id': 0}).sort('uploaded_at', -1).limit(10))
    
    def get_inventory_data(self, days=90):
        """Get comprehensive inventory data with trends"""
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            records = list(self.inventory_db["inventory_records"].find({
                "metadata.date": {"$gte": start_date}
            }))
            
            metals = {}
            total_value = 0
            daily_data = defaultdict(lambda: {'value': 0, 'items': 0, 'metals': {}})
            
            for record in records:
                date = record.get('metadata', {}).get('date', 'Unknown')
                
                for item in record.get('items', []):
                    metal = item.get('metal_name', 'Unknown')
                    qty = item.get('quantity', 0)
                    cost = item.get('total_cost', 0)
                    
                    if metal not in metals:
                        metals[metal] = {
                            'quantity': 0, 
                            'value': 0, 
                            'purchases': 0,
                            'avg_price': 0,
                            'daily_data': []
                        }
                    
                    metals[metal]['quantity'] += qty
                    metals[metal]['value'] += cost
                    metals[metal]['purchases'] += 1
                    
                    total_value += cost
                    daily_data[date]['value'] += cost
                    daily_data[date]['items'] += 1
                    if metal not in daily_data[date]['metals']:
                        daily_data[date]['metals'][metal] = 0
                    daily_data[date]['metals'][metal] += cost
            
            # Calculate trends
            for metal in metals:
                if metals[metal]['quantity'] > 0:
                    metals[metal]['avg_price'] = metals[metal]['value'] / metals[metal]['quantity']
            
            return {
                'total_value': total_value,
                'total_records': len(records),
                'metals': metals,
                'daily_data': dict(daily_data),
                'top_metals': sorted(metals.items(), key=lambda x: x[1]['value'], reverse=True),
                'timeframe_days': days
            }
        except Exception as e:
            return {'total_value': 0, 'metals': {}, 'top_metals': [], 'daily_data': {}, 'error': str(e)}
    
    def get_finance_data(self):
        """Get comprehensive financial data"""
        try:
            statements = list(self.finance_db["bank_statements"].find({}))
            invoices = list(self.finance_db["invoices"].find({}))
            
            total_balance = sum(s.get('balances', {}).get('current_balance', 0) for s in statements)
            total_deposits = sum(s.get('transactions_summary', {}).get('total_deposits', 0) for s in statements)
            total_withdrawals = sum(s.get('transactions_summary', {}).get('total_withdrawals', 0) for s in statements)
            total_invoice_value = sum(i.get('totals', {}).get('total', 0) for i in invoices)
            
            return {
                'bank_balance': total_balance,
                'total_deposits': total_deposits,
                'total_withdrawals': total_withdrawals,
                'invoice_value': total_invoice_value,
                'statements_count': len(statements),
                'invoices_count': len(invoices),
                'cash_flow': total_deposits - total_withdrawals
            }
        except:
            return {
                'bank_balance': 0, 
                'total_deposits': 0, 
                'total_withdrawals': 0,
                'invoice_value': 0,
                'cash_flow': 0
            }
    
    def save_chat(self, user_msg: str, ai_msg: str):
        """Save chat history"""
        self.marketing_db["chat_history"].insert_one({
            'timestamp': datetime.now().isoformat(),
            'user': user_msg,
            'ai': ai_msg
        })
    
    def get_chat_history(self, limit=50):
        """Get recent chat history for context"""
        return list(self.marketing_db["chat_history"]
                   .find({}, {'_id': 0})
                   .sort('timestamp', -1)
                   .limit(limit))

# =============================================================================
# STATE-OF-THE-ART MARKETING AI
# =============================================================================

class StateOfTheArtMarketingAI:
    """Advanced AI Marketing Agent with deep analytical capabilities"""
    
    def __init__(self, db: AdvancedDBManager):
        self.db = db
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            api_key=Config.OPENAI_API_KEY
        )
    
    def chat(self, user_message: str) -> Tuple[str, Optional[Dict]]:
        """
        Main conversational interface - returns response and optional chart data
        Returns: (text_response, chart_data_dict or None)
        """
        try:
            # First, analyze the user query to determine relevant data to fetch
            query_analysis_prompt = PromptTemplate(
                input_variables=["user_message"],
                template="""Analyze the user's query: {user_message}

Determine what data is relevant:
- If related to ads/marketing: facebook_ads
- If related to inventory: inventory_data
- If related to finance: finance_data
- If general or multiple: all

Also, extract any specific parameters like time periods, specific metrics, or filters.

Return JSON:
{{
    "relevant_data": ["facebook_ads", "inventory_data", "finance_data"] or subset,
    "parameters": {{
        "days": number or null,
        "filters": "description or null"
    }},
    "intent": "brief summary of query intent"
}}"""
            )
            
            query_chain = LLMChain(llm=self.llm, prompt=query_analysis_prompt)
            query_response = query_chain.invoke({"user_message": user_message})
            try:
                query_plan = json.loads(query_response['text'])
            except:
                query_plan = {
                    "relevant_data": ["facebook_ads", "inventory_data", "finance_data"],
                    "parameters": {},
                    "intent": user_message
                }
            
            # Fetch only relevant data based on query plan
            fb_ads = self.db.get_facebook_ads() if "facebook_ads" in query_plan["relevant_data"] else []
            days = query_plan["parameters"].get("days", 90)
            inventory = self.db.get_inventory_data(days) if "inventory_data" in query_plan["relevant_data"] else {}
            finance = self.db.get_finance_data() if "finance_data" in query_plan["relevant_data"] else {}
            history = self.db.get_chat_history(10)
            
            # Build intelligent context from fetched data
            context = self._build_intelligent_context(fb_ads, inventory, finance, history)
            
            # Define LangChain prompt template - emphasize valuing user prompt
            chat_prompt = PromptTemplate(
                input_variables=["user_message", "context", "history"],
                template="""You are an elite AI Marketing Strategist for a metal recycling business in Mexico.

USER QUESTION (HIGH PRIORITY - RESPOND DIRECTLY AND PRECISELY TO THIS): {user_message}

BUSINESS INTELLIGENCE CONTEXT (USE ONLY RELEVANT PARTS):
{context}

CONVERSATION HISTORY (Last 10 messages):
{history}

YOUR CAPABILITIES:
1. Deep data analysis across Facebook Ads, Inventory, and Finance
2. Pattern recognition and trend identification
3. Predictive insights and forecasting
4. ROI optimization and budget allocation
5. Campaign performance evaluation
6. Marketing strategy development for any timeframe
7. Seasonal pattern detection
8. Competitive positioning recommendations

YOUR TASK:
- Prioritize the user's exact question above all - provide the most relevant, targeted response
- Fetch and use only data directly related to the query
- Analyze specifically for the user's intent
- Make sophisticated connections between different data sources when relevant
- Calculate precise metrics (ROI, conversion rates, efficiency scores)
- Identify patterns, trends, and anomalies related to the query
- Give specific, actionable recommendations tailored to the prompt
- When discussing timeframes, provide detailed day-by-day or week-by-week plans if requested
- Be conversational yet professional
- Support claims with actual numbers from the data

RESPONSE GUIDELINES:
1. Start with a direct, clear answer to the user's question
2. Support with specific data points and metrics most relevant to the query
3. Reveal insights and patterns discovered in the data that match the user's intent
4. Make intelligent connections only if they enhance the response to the query
5. Provide 2-3 concrete, prioritized recommendations directly addressing the prompt
6. When asked about plans, provide detailed, realistic strategies
7. Be honest about data limitations if any
8. Ensure the response is concise and focused on the most related information

Generate your expert analysis:"""
            )
            
            # Create LangChain chain
            chat_chain = LLMChain(llm=self.llm, prompt=chat_prompt)
            
            # Invoke chain
            response = chat_chain.invoke({
                "user_message": user_message,
                "context": context,
                "history": self._format_history(history)
            })
            answer = response['text']
            
            # Save to history
            self.db.save_chat(user_message, answer)
            
            return answer, None
            
        except Exception as e:
            return f"I apologize, but I encountered an issue: {str(e)}\n\nPlease try rephrasing your question or ask about:\n- Marketing ROI and performance\n- Campaign recommendations\n- Budget allocation strategies\n- Inventory-based marketing opportunities", None
    
    def _build_intelligent_context(self, fb_ads: List, inventory: Dict, finance: Dict, history: List) -> str:
        """Build comprehensive business intelligence context"""
        
        parts = []
        
        # Facebook Ads Intelligence
        if fb_ads:
            latest = fb_ads[0]
            analysis = latest.get('analysis', {})
            totals = analysis.get('totals', {})
            trends = analysis.get('trends', {})
            
            parts.append(f"""
═══════════════════════════════════════
📱 FACEBOOK ADS INTELLIGENCE
═══════════════════════════════════════
Total Investment: ${totals.get('total_spend', 0):,.2f}
Total Reach: {totals.get('total_impressions', 0):,} impressions
Engagement: {totals.get('total_clicks', 0):,} clicks
Performance: {totals.get('avg_ctr', 0):.2f}% CTR | ${totals.get('avg_cpc', 0):.2f} CPC
Active Campaigns: {len(analysis.get('campaigns', []))}

🎯 Best Performer: {analysis.get('best_campaign', {}).get('name', 'N/A')}
   Reason: {analysis.get('best_campaign', {}).get('reason', 'N/A')}

⚠️  Needs Attention: {analysis.get('worst_campaign', {}).get('name', 'N/A')}
   Issue: {analysis.get('worst_campaign', {}).get('reason', 'N/A')}

📊 Trends Detected:
   • Engagement: {trends.get('engagement_trend', 'Unknown')}
   • Efficiency: {trends.get('spend_efficiency', 'Unknown')}
   • Audience: {trends.get('audience_response', 'Analyzing')}
""")
            
            # Campaign details
            if analysis.get('campaigns'):
                parts.append("\n📋 CAMPAIGN BREAKDOWN:")
                for camp in analysis['campaigns'][:5]:
                    parts.append(f"""
   • {camp.get('name', 'Unknown')}
     Spend: ${camp.get('spend', 0):,.2f} | Clicks: {camp.get('clicks', 0):,}
     CTR: {camp.get('ctr', 0):.2f}% | CPC: ${camp.get('cpc', 0):.2f}""")
        
        # Inventory Intelligence
        parts.append(f"""

═══════════════════════════════════════
📦 INVENTORY INTELLIGENCE ({inventory.get('timeframe_days', 90)} Days)
═══════════════════════════════════════
Total Revenue: ${inventory.get('total_value', 0):,.2f}
Purchase Transactions: {inventory.get('total_records', 0)}
Unique Metals: {len(inventory.get('metals', {}))}
Avg Transaction Value: ${inventory.get('total_value', 0) / max(inventory.get('total_records', 1), 1):,.2f}

🥇 TOP 5 REVENUE METALS:""")
        
        for i, (metal, data) in enumerate(inventory.get('top_metals', [])[:5], 1):
            market_share = (data['value'] / inventory.get('total_value', 1)) * 100
            parts.append(f"""
   {i}. {metal}
      Revenue: ${data['value']:,.2f} ({market_share:.1f}% of total)
      Volume: {data['quantity']:.2f} kg
      Transactions: {data['purchases']}
      Avg Price: ${data.get('avg_price', 0):.2f}/kg""")
        
        # Financial Intelligence
        parts.append(f"""

═══════════════════════════════════════
💰 FINANCIAL INTELLIGENCE
═══════════════════════════════════════
Bank Balance: ${finance.get('bank_balance', 0):,.2f}
Total Deposits: ${finance.get('total_deposits', 0):,.2f}
Total Withdrawals: ${finance.get('total_withdrawals', 0):,.2f}
Net Cash Flow: ${finance.get('cash_flow', 0):,.2f}
Pending Invoices: ${finance.get('invoice_value', 0):,.2f}
""")
        
        # Strategic Insights
        if fb_ads and inventory.get('total_value', 0) > 0:
            ad_spend = fb_ads[0].get('analysis', {}).get('totals', {}).get('total_spend', 0)
            inv_value = inventory.get('total_value', 0)
            
            if ad_spend > 0:
                roi = ((inv_value - ad_spend) / ad_spend) * 100
                roas = inv_value / ad_spend if ad_spend > 0 else 0
                
                parts.append(f"""
═══════════════════════════════════════
🎯 STRATEGIC PERFORMANCE METRICS
═══════════════════════════════════════
Marketing Investment: ${ad_spend:,.2f}
Revenue Generated: ${inv_value:,.2f}
Net Profit: ${inv_value - ad_spend:,.2f}
ROI: {roi:+.1f}% {'✅ PROFITABLE' if roi > 0 else '❌ LOSING MONEY'}
ROAS: {roas:.2f}x (Return on Ad Spend)
Break-even: {"Achieved" if roi > 0 else f"Need ${ad_spend - inv_value:,.2f} more revenue"}
""")
        
        return "\n".join(parts)
    
    def _format_history(self, history: List) -> str:
        """Format chat history for context"""
        if not history:
            return "No previous conversation"
        
        formatted = []
        for msg in reversed(history[-5:]):  # Last 5 messages
            formatted.append(f"User: {msg.get('user', '')[:100]}")
            formatted.append(f"AI: {msg.get('ai', '')[:150]}\n")
        
        return "\n".join(formatted)