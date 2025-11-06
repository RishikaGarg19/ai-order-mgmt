
# Version 4
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Visualizer:
    def __init__(self, df):
        self.df = df
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'purple': '#9467bd',
            'brown': '#8c564b',
            'pink': '#e377c2'
        }
    
    def format_large_number(self, value):
        """Format large numbers with K, M, B suffixes"""
        if pd.isna(value) or value == 0:
            return "0"
        
        if abs(value) >= 1_000_000_000:
            return f"{value/1_000_000_000:.1f}B"
        elif abs(value) >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            return f"{value/1_000:.1f}K"
        else:
            return f"{value:,.0f}"

    def format_currency(self, value):
        """Format currency values with K, M, B suffixes"""
        if pd.isna(value) or value == 0:
            return "$0"
        
        if abs(value) >= 1_000_000_000:
            return f"${value/1_000_000_000:.1f}B"
        elif abs(value) >= 1_000_000:
            return f"${value/1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            return f"${value/1_000:.1f}K"
        else:
            return f"${value:,.0f}"
    
    def create_visualizations(self, query, analysis_result):
        """Create basic visualizations - kept for backward compatibility"""
        self.create_enhanced_visualizations(query, analysis_result)
    
    def create_enhanced_visualizations(self, query, analysis_result):
        """Create enhanced interactive visualizations based on query and analysis"""
        
        try:
            query_lower = query.lower()
            
            # Route to appropriate enhanced visualization based on query content
            if any(keyword in query_lower for keyword in ['may 2025', 'current month', 'orders for']):
                self._create_enhanced_current_month_visualizations(query)
            
            elif any(keyword in query_lower for keyword in ['revenue', 'accomplished', 'completed']):
                self._create_enhanced_revenue_visualizations(query)
            
            elif any(keyword in query_lower for keyword in ['delay', 'challenges', 'top']):
                self._create_enhanced_delay_visualizations(query)
            
            elif any(keyword in query_lower for keyword in ['impact', 'impacted']):
                self._create_enhanced_impact_visualizations(query)
            
            elif any(keyword in query_lower for keyword in ['customer tier', 'by customer']):
                self._create_enhanced_customer_tier_visualizations(query)
            
            elif any(keyword in query_lower for keyword in ['average order value', 'product category']):
                self._create_enhanced_avg_order_value_visualizations(query)
            
            elif any(keyword in query_lower for keyword in ['prioritization', 'priority', 'ai', 'ml']):
                if analysis_result.get("type") == "ai_prioritization":
                    self._create_enhanced_ai_prioritization_charts(analysis_result)
                else:
                    self._create_enhanced_general_visualizations(query)
            
            else:
                self._create_enhanced_general_visualizations(query)
        
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
            self._create_fallback_visualization(query)
    
    def _create_enhanced_current_month_visualizations(self, query):
        """Create enhanced visualizations for current month orders"""
        st.subheader("May 2025 Order Analytics")
        
        # Use May 2025 data
        may_2025_mask = (
            (self.df['order_date'].dt.month == 5) & 
            (self.df['order_date'].dt.year == 2025)
        )
        may_orders = self.df[may_2025_mask]
        
        if len(may_orders) == 0:
            st.warning("No orders found for May 2025")
            return
        
        # Enhanced metrics with trend indicators
        unique_orders = may_orders['order_id'].nunique()
        total_value = may_orders['order_value_usd'].sum()
        avg_value = may_orders['order_value_usd'].mean()
        
        # Compare with previous month (April 2025)
        april_2025_mask = (
            (self.df['order_date'].dt.month == 4) & 
            (self.df['order_date'].dt.year == 2025)
        )
        april_orders = self.df[april_2025_mask]
        
        # Calculate growth rates
        if len(april_orders) > 0:
            april_orders_count = april_orders['order_id'].nunique()
            april_value = april_orders['order_value_usd'].sum()
            
            order_growth = ((unique_orders - april_orders_count) / april_orders_count * 100) if april_orders_count > 0 else 0
            value_growth = ((total_value - april_value) / april_value * 100) if april_value > 0 else 0
        else:
            order_growth = 0
            value_growth = 0
        
        # Display metrics with trends
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Unique Orders", 
                self.format_large_number(unique_orders),
                delta=f"{order_growth:+.1f}%" if order_growth != 0 else None
            )
        with col2:
            st.metric(
                "Total Value", 
                self.format_currency(total_value),
                delta=f"{value_growth:+.1f}%" if value_growth != 0 else None
            )
        with col3:
            st.metric("Avg Order Value", self.format_currency(avg_value))
        
        # Enhanced Order Status with drill-down capability
        if 'order_status' in self.df.columns:
            status_counts = may_orders['order_status'].value_counts()
            
            # Interactive pie chart with hover details
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Order Status Distribution - May 2025",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4  # Donut chart for modern look
            )
            
            # Enhanced hover template
            fig_status.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>' +
                            'Orders: %{value}<br>' +
                            'Percentage: %{percent}<br>' +
                            '<extra></extra>'
            )
            
            fig_status.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family="Segoe UI",
                title_font_size=16,
                title_font_color='#2C3E50',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_status, use_container_width=True)
        
        # Enhanced Customer Analysis with cross-filtering
        if 'customer_type' in self.df.columns:
            customer_data = may_orders.groupby('customer_type').agg({
                'order_id': 'nunique',
                'order_value_usd': ['sum', 'mean']
            }).round(2)
            
            customer_data.columns = ['Order_Count', 'Total_Revenue', 'Avg_Order_Value']
            customer_data = customer_data.reset_index()
            
            # Interactive bar chart with secondary metrics
            fig_customer = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Order Count by Customer Type', 'Revenue by Customer Type'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Order count
            fig_customer.add_trace(
                go.Bar(
                    x=customer_data['customer_type'],
                    y=customer_data['Order_Count'],
                    name='Order Count',
                    marker_color=self.colors['primary'],
                    text=customer_data['Order_Count'],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Orders: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Revenue
            fig_customer.add_trace(
                go.Bar(
                    x=customer_data['customer_type'],
                    y=customer_data['Total_Revenue'],
                    name='Revenue',
                    marker_color=self.colors['success'],
                    text=[self.format_currency(x) for x in customer_data['Total_Revenue']],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Revenue: %{text}<extra></extra>'
                ),
                row=1, col=2
            )
            
            fig_customer.update_layout(
                title_text="Customer Analysis - May 2025",
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family="Segoe UI"
            )
            st.plotly_chart(fig_customer, use_container_width=True)
        
        # Enhanced Daily Trend with forecasting indicators
        if 'order_date' in self.df.columns:
            daily_data = may_orders.groupby(may_orders['order_date'].dt.day).agg({
                'order_id': 'nunique',
                'order_value_usd': 'sum'
            }).reset_index()
            daily_data.columns = ['Day', 'Orders', 'Revenue']
            
            # Create dual-axis chart
            fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Order count line
            fig_trend.add_trace(
                go.Scatter(
                    x=daily_data['Day'],
                    y=daily_data['Orders'],
                    mode='lines+markers',
                    name='Orders',
                    line=dict(color=self.colors['primary'], width=3),
                    marker=dict(size=8),
                    hovertemplate='Day %{x}<br>Orders: %{y}<extra></extra>'
                ),
                secondary_y=False,
            )
            
            # Revenue bars
            fig_trend.add_trace(
                go.Bar(
                    x=daily_data['Day'],
                    y=daily_data['Revenue'],
                    name='Revenue',
                    marker_color=self.colors['success'],
                    opacity=0.7,
                    hovertemplate='Day %{x}<br>Revenue: $%{y:,.0f}<extra></extra>'
                ),
                secondary_y=True,
            )
            
            # Add trend line
            if len(daily_data) > 1:
                z = np.polyfit(daily_data['Day'], daily_data['Orders'], 1)
                p = np.poly1d(z)
                fig_trend.add_trace(
                    go.Scatter(
                        x=daily_data['Day'],
                        y=p(daily_data['Day']),
                        mode='lines',
                        name='Trend',
                        line=dict(color=self.colors['danger'], dash='dash', width=2),
                        hovertemplate='Trend Line<extra></extra>'
                    ),
                    secondary_y=False,
                )
            
            fig_trend.update_xaxes(title_text="Day of Month")
            fig_trend.update_yaxes(title_text="Number of Orders", secondary_y=False)
            fig_trend.update_yaxes(title_text="Revenue ($)", secondary_y=True)
            
            fig_trend.update_layout(
                title_text="Daily Performance Trend - May 2025",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family="Segoe UI",
                hovermode='x unified'
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    
    def _create_enhanced_revenue_visualizations(self, query):
        """Create enhanced visualizations for revenue analysis"""
        st.subheader("Revenue Performance Analytics")
        
        # Use May 2025 data
        may_2025_mask = (
            (self.df['order_date'].dt.month == 5) & 
            (self.df['order_date'].dt.year == 2025)
        )
        may_orders = self.df[may_2025_mask]
        
        # Filter completed orders
        if 'order_status' in self.df.columns:
            completed_orders = may_orders[
                may_orders['order_status'].isin(['Completed', 'Delivered', 'Shipped'])
            ]
        else:
            completed_orders = may_orders
        
        if len(completed_orders) == 0:
            st.warning("No completed orders found for May 2025")
            return
        
        # Enhanced Revenue Metrics
        total_revenue = completed_orders['order_value_usd'].sum()
        avg_revenue = completed_orders['order_value_usd'].mean()
        order_count = len(completed_orders)
        median_revenue = completed_orders['order_value_usd'].median()
        
        # Revenue distribution metrics
        q75 = completed_orders['order_value_usd'].quantile(0.75)
        q25 = completed_orders['order_value_usd'].quantile(0.25)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Revenue", self.format_currency(total_revenue))
        with col2:
            st.metric("Completed Orders", self.format_large_number(order_count))
        with col3:
            st.metric("Avg Order Value", self.format_currency(avg_revenue))
        with col4:
            st.metric("Median Order Value", self.format_currency(median_revenue))
        
        # Enhanced Revenue Distribution Analysis
        if 'customer_type' in self.df.columns:
            customer_revenue = completed_orders.groupby('customer_type').agg({
                'order_value_usd': ['sum', 'mean', 'count', 'std']
            }).round(2)
            
            customer_revenue.columns = ['Total_Revenue', 'Avg_Revenue', 'Order_Count', 'Revenue_Std']
            customer_revenue = customer_revenue.reset_index()
            customer_revenue['Revenue_Share'] = (customer_revenue['Total_Revenue'] / total_revenue * 100).round(1)
            
            # Interactive sunburst chart for hierarchical revenue view
            fig_sunburst = px.sunburst(
                completed_orders,
                path=['customer_type', 'customer_tier'] if 'customer_tier' in completed_orders.columns else ['customer_type'],
                values='order_value_usd',
                title="Revenue Hierarchy - Customer Type & Tier",
                color='order_value_usd',
                color_continuous_scale='Viridis'
            )
            
            fig_sunburst.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family="Segoe UI"
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)
        
        # Enhanced Product Category Analysis
        if 'product_category' in self.df.columns:
            category_analysis = completed_orders.groupby('product_category').agg({
                'order_value_usd': ['sum', 'mean', 'count'],
                'profit_margin_percent': 'mean' if 'profit_margin_percent' in completed_orders.columns else lambda x: 0
            }).round(2)
            
            # Flatten column names
            category_analysis.columns = ['Total_Revenue', 'Avg_Revenue', 'Order_Count', 'Avg_Margin']
            category_analysis = category_analysis.reset_index()
            
            # Bubble chart showing revenue vs margin vs count
            fig_bubble = px.scatter(
                category_analysis,
                x='Avg_Revenue',
                y='Avg_Margin' if 'profit_margin_percent' in completed_orders.columns else 'Order_Count',
                size='Total_Revenue',
                color='Total_Revenue',
                hover_name='product_category',
                title="Product Performance Matrix",
                labels={
                    'Avg_Revenue': 'Average Order Value ($)',
                    'Avg_Margin': 'Average Profit Margin (%)',
                    'Order_Count': 'Order Count'
                },
                color_continuous_scale='Turbo'
            )
            
            fig_bubble.update_traces(
                hovertemplate='<b>%{hovertext}</b><br>' +
                            'Avg Order Value: $%{x:,.0f}<br>' +
                            'Orders: %{marker.size:,.0f}<br>' +
                            '<extra></extra>'
            )
            
            fig_bubble.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family="Segoe UI"
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Revenue Trend with Forecasting
        if len(completed_orders) > 1:
            daily_revenue = completed_orders.groupby(completed_orders['order_date'].dt.day)['order_value_usd'].sum().reset_index()
            daily_revenue.columns = ['Day', 'Revenue']
            
            # Create trend with moving average
            if len(daily_revenue) >= 3:
                daily_revenue['MA_3'] = daily_revenue['Revenue'].rolling(window=3, center=True).mean()
            
            fig_revenue_trend = go.Figure()
            
            # Actual revenue
            fig_revenue_trend.add_trace(
                go.Scatter(
                    x=daily_revenue['Day'],
                    y=daily_revenue['Revenue'],
                    mode='lines+markers',
                    name='Daily Revenue',
                    line=dict(color=self.colors['primary'], width=3),
                    marker=dict(size=8),
                    fill='tonexty' if len(daily_revenue) > 1 else None,
                    fillcolor='rgba(31, 119, 180, 0.2)'
                )
            )
            
            # Moving average if available
            if len(daily_revenue) >= 3:
                fig_revenue_trend.add_trace(
                    go.Scatter(
                        x=daily_revenue['Day'],
                        y=daily_revenue['MA_3'],
                        mode='lines',
                        name='3-Day Moving Average',
                        line=dict(color=self.colors['danger'], dash='dash', width=2)
                    )
                )
            
            fig_revenue_trend.update_layout(
                title="Daily Revenue Trend - May 2025",
                xaxis_title="Day of Month",
                yaxis_title="Revenue ($)",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family="Segoe UI",
                hovermode='x unified'
            )
            st.plotly_chart(fig_revenue_trend, use_container_width=True)
    
    def _create_enhanced_delay_visualizations(self, query):
        """Create enhanced visualizations for delay analysis"""
        st.subheader("Delay Performance Analytics")
        
        if 'is_delayed' not in self.df.columns:
            st.warning("No delay data available")
            return
        
        delayed_orders = self.df[self.df['is_delayed'] == True]
        
        if len(delayed_orders) == 0:
            st.info("No delayed orders found")
            return
        
        # Enhanced Delay Metrics
        total_delayed = len(delayed_orders)
        avg_delay_days = delayed_orders['delay_days'].mean() if 'delay_days' in self.df.columns else 0
        delay_rate = (total_delayed / len(self.df)) * 100
        max_delay = delayed_orders['delay_days'].max() if 'delay_days' in self.df.columns else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Delayed Orders", self.format_large_number(total_delayed))
        with col2:
            st.metric("Delay Rate", f"{delay_rate:.1f}%")
        with col3:
            st.metric("Avg Delay Days", f"{avg_delay_days:.1f}")
        with col4:
            st.metric("Max Delay Days", f"{max_delay:.0f}")
        
        # Enhanced Delay Category Analysis with Impact
        if 'primary_delay_category' in self.df.columns:
            delay_analysis = delayed_orders.groupby('primary_delay_category').agg({
                'order_id': 'count',
                'delay_days': 'mean',
                'order_value_usd': 'sum' if 'order_value_usd' in delayed_orders.columns else 'count',
                'revenue_impact_usd': 'sum' if 'revenue_impact_usd' in delayed_orders.columns else 'count'
            }).round(2)
            
            delay_analysis.columns = ['Order_Count', 'Avg_Delay_Days', 'Total_Order_Value', 'Revenue_Impact']
            delay_analysis = delay_analysis.reset_index().sort_values('Order_Count', ascending=False)
            
            # Enhanced waterfall chart for delay impact
            fig_waterfall = go.Figure(go.Waterfall(
                name="Delay Impact",
                orientation="v",
                measure=["relative"] * len(delay_analysis),
                x=delay_analysis['primary_delay_category'],
                y=delay_analysis['Order_Count'],
                text=[f"{count} orders" for count in delay_analysis['Order_Count']],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": self.colors['danger']}},
                decreasing={"marker": {"color": self.colors['success']}},
                totals={"marker": {"color": self.colors['info']}}
            ))
            
            fig_waterfall.update_layout(
                title="Delay Category Impact Analysis",
                xaxis_title="Delay Category",
                yaxis_title="Number of Orders",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family="Segoe UI"
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # Delay Severity Heat Map
        if 'delay_days' in self.df.columns and 'order_value_usd' in self.df.columns:
            # Create delay severity bins
            delayed_orders_copy = delayed_orders.copy()
            delayed_orders_copy['Delay_Severity'] = pd.cut(
                delayed_orders_copy['delay_days'], 
                bins=[0, 7, 14, 30, float('inf')], 
                labels=['1-7 days', '8-14 days', '15-30 days', '30+ days']
            )
            
            delayed_orders_copy['Value_Category'] = pd.cut(
                delayed_orders_copy['order_value_usd'], 
                bins=[0, 100000, 500000, 1000000, float('inf')], 
                labels=['<$100K', '$100K-$500K', '$500K-$1M', '$1M+']
            )
            
            # Create heat map data
            heatmap_data = delayed_orders_copy.groupby(['Delay_Severity', 'Value_Category']).size().unstack(fill_value=0)
            
            fig_heatmap = px.imshow(
                heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale='Reds',
                title="Delay Severity vs Order Value Heat Map",
                labels=dict(x="Order Value Category", y="Delay Severity", color="Order Count")
            )
            
            # Add text annotations
            for i, row in enumerate(heatmap_data.values):
                for j, value in enumerate(row):
                    if value > 0:
                        fig_heatmap.add_annotation(
                            x=j, y=i, text=str(int(value)),
                            showarrow=False, font=dict(color="white" if value > heatmap_data.values.max()/2 else "black")
                        )
            
            fig_heatmap.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family="Segoe UI"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    def _create_enhanced_impact_visualizations(self, query):
        """Create enhanced visualizations for revenue impact analysis"""
        st.subheader("Revenue Impact Analytics")
        
        if 'revenue_impact_usd' not in self.df.columns:
            st.warning("No revenue impact data available")
            return
        
        impact_orders = self.df[self.df['revenue_impact_usd'] > 0]
        
        if len(impact_orders) == 0:
            st.info("No revenue impact found")
            return
        
        # Enhanced Impact Metrics
        total_impact = impact_orders['revenue_impact_usd'].sum()
        avg_impact = impact_orders['revenue_impact_usd'].mean()
        impact_orders_count = len(impact_orders)
        median_impact = impact_orders['revenue_impact_usd'].median()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Impact", self.format_currency(total_impact))
        with col2:
            st.metric("Affected Orders", self.format_large_number(impact_orders_count))
        with col3:
            st.metric("Avg Impact", self.format_currency(avg_impact))
        with col4:
            st.metric("Median Impact", self.format_currency(median_impact))
        
        # Enhanced Pareto Analysis for Impact Categories
        if 'primary_delay_category' in self.df.columns:
            impact_by_category = impact_orders.groupby('primary_delay_category')['revenue_impact_usd'].sum().sort_values(ascending=False)
            
            # Calculate cumulative percentage
            cumsum = impact_by_category.cumsum()
            cumulative_pct = (cumsum / total_impact * 100).round(1)
            
            # Create Pareto chart
            fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Bars for impact
            fig_pareto.add_trace(
                go.Bar(
                    x=impact_by_category.index,
                    y=impact_by_category.values,
                    name='Revenue Impact',
                    marker_color=self.colors['danger'],
                    text=[self.format_currency(x) for x in impact_by_category.values],
                    textposition='outside'
                ),
                secondary_y=False,
            )
            
            # Line for cumulative percentage
            fig_pareto.add_trace(
                go.Scatter(
                    x=impact_by_category.index,
                    y=cumulative_pct.values,
                    mode='lines+markers',
                    name='Cumulative %',
                    line=dict(color=self.colors['warning'], width=3),
                    marker=dict(size=8)
                ),
                secondary_y=True,
            )
            
            # Add 80% line
            fig_pareto.add_hline(
                y=80, line_dash="dash", line_color="red",
                annotation_text="80% Rule", secondary_y=True
            )
            
            fig_pareto.update_xaxes(title_text="Delay Category")
            fig_pareto.update_yaxes(title_text="Revenue Impact ($)", secondary_y=False)
            fig_pareto.update_yaxes(title_text="Cumulative Percentage (%)", secondary_y=True)
            
            fig_pareto.update_layout(
                title_text="Pareto Analysis - Revenue Impact by Delay Category",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family="Segoe UI"
            )
            st.plotly_chart(fig_pareto, use_container_width=True)
    
    def _create_enhanced_customer_tier_visualizations(self, query):
        """Create enhanced visualizations for customer tier analysis"""
        st.subheader("Customer Tier Performance Analytics")
        
        if 'customer_tier' not in self.df.columns:
            st.warning("No customer tier data available")
            return
        
        # Use May 2025 data
        may_2025_mask = (
            (self.df['order_date'].dt.month == 5) & 
            (self.df['order_date'].dt.year == 2025)
        )
        may_orders = self.df[may_2025_mask]
        
        if len(may_orders) == 0:
            st.warning("No orders found for May 2025")
            return
        
        # Enhanced Tier Analysis
        tier_analysis = may_orders.groupby('customer_tier').agg({
            'order_id': 'nunique',
            'order_value_usd': ['sum', 'mean', 'median'],
            'is_delayed': 'mean' if 'is_delayed' in may_orders.columns else lambda x: 0,
            'customer_satisfaction': 'mean' if 'customer_satisfaction' in may_orders.columns else lambda x: 0
        }).round(2)
        
        # Flatten column names
        tier_analysis.columns = ['Order_Count', 'Total_Revenue', 'Avg_Order_Value', 'Median_Order_Value', 'Delay_Rate', 'Avg_Satisfaction']
        tier_analysis = tier_analysis.reset_index()
        
        # Enhanced metrics display
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Orders", self.format_large_number(len(may_orders)))
        with col2:
            st.metric("Customer Tiers", len(tier_analysis))
        
        # Enhanced Multi-dimensional Analysis
        fig_multi = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Orders by Tier', 'Revenue by Tier', 'Average Order Value', 'Performance Metrics'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Orders by tier
        fig_multi.add_trace(
            go.Bar(
                x=tier_analysis['customer_tier'],
                y=tier_analysis['Order_Count'],
                name='Orders',
                marker_color=self.colors['primary'],
                text=tier_analysis['Order_Count'],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Revenue pie chart
        fig_multi.add_trace(
            go.Pie(
                labels=tier_analysis['customer_tier'],
                values=tier_analysis['Total_Revenue'],
                name='Revenue',
                textinfo='label+percent',
                textposition='inside'
            ),
            row=1, col=2
        )
        
        # Average order value
        fig_multi.add_trace(
            go.Bar(
                x=tier_analysis['customer_tier'],
                y=tier_analysis['Avg_Order_Value'],
                name='Avg Order Value',
                marker_color=self.colors['success'],
                text=[self.format_currency(x) for x in tier_analysis['Avg_Order_Value']],
                textposition='outside'
            ),
            row=2, col=1
        )
        
        # Performance scatter (if data available)
        if 'customer_satisfaction' in may_orders.columns and 'is_delayed' in may_orders.columns:
            fig_multi.add_trace(
                go.Scatter(
                    x=tier_analysis['Delay_Rate'] * 100,
                    y=tier_analysis['Avg_Satisfaction'],
                    mode='markers+text',
                    text=tier_analysis['customer_tier'],
                    textposition='top center',
                    marker=dict(
                        size=tier_analysis['Total_Revenue'] / tier_analysis['Total_Revenue'].max() * 50,
                        color=tier_analysis['Total_Revenue'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Performance'
                ),
                row=2, col=2
            )
        
        fig_multi.update_layout(
            title_text="Customer Tier Comprehensive Analysis - May 2025",
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_family="Segoe UI",
            height=800
        )
        st.plotly_chart(fig_multi, use_container_width=True)
    
    def _create_enhanced_avg_order_value_visualizations(self, query):
        """Create enhanced visualizations for average order value analysis"""
        st.subheader("Average Order Value Performance Analytics")
        
        if 'product_category' not in self.df.columns or 'order_value_usd' not in self.df.columns:
            st.warning("Required data not available")
            return
        
        # Enhanced AOV Analysis
        aov_analysis = self.df.groupby('product_category').agg({
            'order_value_usd': ['mean', 'count', 'sum', 'std', 'median'],
            'profit_margin_percent': 'mean' if 'profit_margin_percent' in self.df.columns else lambda x: 0
        }).round(2)
        
        aov_analysis.columns = ['Avg_Order_Value', 'Order_Count', 'Total_Revenue', 'AOV_Std', 'Median_AOV', 'Avg_Margin']
        aov_analysis = aov_analysis.reset_index()
        overall_avg = self.df['order_value_usd'].mean()
        
        # Calculate performance vs benchmark
        aov_analysis['Performance_vs_Avg'] = ((aov_analysis['Avg_Order_Value'] / overall_avg - 1) * 100).round(1)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Avg AOV", self.format_currency(overall_avg))
        with col2:
            st.metric("Product Categories", len(aov_analysis))
        with col3:
            best_category = aov_analysis.loc[aov_analysis['Avg_Order_Value'].idxmax(), 'product_category']
            st.metric("Best Performing Category", best_category)
        
        # Enhanced Performance Matrix
        fig_matrix = px.scatter(
            aov_analysis,
            x='Order_Count',
            y='Avg_Order_Value',
            size='Total_Revenue',
            color='Avg_Margin' if 'profit_margin_percent' in self.df.columns else 'Performance_vs_Avg',
            hover_name='product_category',
            title="Product Category Performance Matrix",
            labels={
                'Order_Count': 'Number of Orders',
                'Avg_Order_Value': 'Average Order Value ($)',
                'Avg_Margin': 'Average Profit Margin (%)',
                'Performance_vs_Avg': 'Performance vs Average (%)'
            },
            color_continuous_scale='RdYlGn'
        )
        
        # Add quadrant lines
        fig_matrix.add_vline(x=aov_analysis['Order_Count'].median(), line_dash="dash", line_color="gray")
        fig_matrix.add_hline(y=overall_avg, line_dash="dash", line_color="gray")
        
        # Add annotations for quadrants
        fig_matrix.add_annotation(
            x=aov_analysis['Order_Count'].max() * 0.8,
            y=aov_analysis['Avg_Order_Value'].max() * 0.9,
            text="Stars<br>(High Volume, High Value)",
            showarrow=False, bgcolor="lightgreen", opacity=0.7
        )
        
        fig_matrix.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_family="Segoe UI"
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Enhanced Comparison Chart
        fig_comparison = go.Figure()
        
        # Sort by average order value
        aov_sorted = aov_analysis.sort_values('Avg_Order_Value', ascending=True)
        
        # Add bars with color coding
        colors = ['green' if x > 0 else 'red' for x in aov_sorted['Performance_vs_Avg']]
        
        fig_comparison.add_trace(
            go.Bar(
                y=aov_sorted['product_category'],
                x=aov_sorted['Avg_Order_Value'],
                orientation='h',
                marker_color=colors,
                text=[self.format_currency(x) for x in aov_sorted['Avg_Order_Value']],
                textposition='outside',
                name='Average Order Value'
            )
        )
        
        # Add overall average line
        fig_comparison.add_vline(
            x=overall_avg, 
            line_dash="dash", 
            line_color="blue",
            annotation_text=f"Overall Avg: {self.format_currency(overall_avg)}"
        )
        
        fig_comparison.update_layout(
            title="Average Order Value by Product Category",
            xaxis_title="Average Order Value ($)",
            yaxis_title="Product Category",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_family="Segoe UI",
            height=400 + len(aov_analysis) * 30
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    def _create_enhanced_ai_prioritization_charts(self, result):
        """Create enhanced visualizations for AI prioritization results"""
        
        if 'top_priorities' not in result or len(result['top_priorities']) == 0:
            st.warning("No prioritization data available for visualization")
            return
        
        st.subheader("AI Prioritization Intelligence Dashboard")
        
        top_priorities = result['top_priorities']
        
        # Enhanced Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Orders", self.format_large_number(result.get('total_orders', 0)))
        with col2:
            st.metric("High Priority", self.format_large_number(result.get('high_priority_count', 0)))
        with col3:
            st.metric("Critical Risk", self.format_large_number(result.get('critical_risk_count', 0)))
        with col4:
            avg_risk = top_priorities['predicted_delay_probability'].mean() if 'predicted_delay_probability' in top_priorities.columns else 0
            st.metric("Avg AI Risk", f"{avg_risk:.1%}")
        
        # Enhanced Priority Distribution with Risk Overlay
        if 'priority_score' in top_priorities.columns and 'predicted_delay_probability' in top_priorities.columns:
            fig_priority = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Priority Score Distribution', 'AI Risk vs Priority Score'),
                specs=[[{"type": "histogram"}, {"type": "scatter"}]]
            )
            
            # Priority distribution
            fig_priority.add_trace(
                go.Histogram(
                    x=top_priorities['priority_score'],
                    nbinsx=20,
                    name='Priority Score',
                    marker_color=self.colors['primary'],
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Risk vs Priority scatter
            fig_priority.add_trace(
                go.Scatter(
                    x=top_priorities['priority_score'],
                    y=top_priorities['predicted_delay_probability'],
                    mode='markers',
                    marker=dict(
                        size=top_priorities['order_value_usd'] / top_priorities['order_value_usd'].max() * 30 if 'order_value_usd' in top_priorities.columns else 10,
                        color=top_priorities['customer_tier'].astype('category').cat.codes if 'customer_tier' in top_priorities.columns else 'blue',
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Customer Tier")
                    ),
                    name='Orders',
                    hovertemplate='<b>Order %{text}</b><br>' +
                                'Priority: %{x:.1f}<br>' +
                                'AI Risk: %{y:.1%}<br>' +
                                '<extra></extra>',
                    text=top_priorities['order_id'] if 'order_id' in top_priorities.columns else range(len(top_priorities))
                ),
                row=1, col=2
            )
            
            fig_priority.update_layout(
                title_text="AI Prioritization Analysis",
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family="Segoe UI"
            )
            st.plotly_chart(fig_priority, use_container_width=True)
    
    def _create_enhanced_general_visualizations(self, query):
        """Create enhanced general visualizations based on available data"""
        st.subheader("General Business Analytics")
        
        # Enhanced Dataset Overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Orders", self.format_large_number(len(self.df)))
        with col2:
            if 'order_value_usd' in self.df.columns:
                total_value = self.df['order_value_usd'].sum()
                st.metric("Total Value", self.format_currency(total_value))
        with col3:
            if 'order_date' in self.df.columns:
                date_range = (self.df['order_date'].max() - self.df['order_date'].min()).days
                st.metric("Date Range", f"{date_range} days")
        
        # Enhanced Status Overview with Trends
        if 'order_status' in self.df.columns:
            status_counts = self.df['order_status'].value_counts()
            
            # Enhanced donut chart
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Overall Order Status Distribution",
                hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_status.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Orders: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            
            fig_status.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family="Segoe UI",
                annotations=[dict(text='Order<br>Status', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            st.plotly_chart(fig_status, use_container_width=True)
        
        # Enhanced Monthly Trend with Seasonality
        if 'order_date' in self.df.columns:
            monthly_data = self.df.groupby(self.df['order_date'].dt.to_period('M')).agg({
                'order_id': 'nunique',
                'order_value_usd': 'sum' if 'order_value_usd' in self.df.columns else 'count'
            }).reset_index()
            
            monthly_data.columns = ['Month', 'Orders', 'Revenue']
            monthly_data['Month'] = monthly_data['Month'].astype(str)
            
            # Enhanced trend with dual axis
            fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_trend.add_trace(
                go.Scatter(
                    x=monthly_data['Month'],
                    y=monthly_data['Orders'],
                    mode='lines+markers',
                    name='Orders',
                    line=dict(color=self.colors['primary'], width=3),
                    marker=dict(size=8)
                ),
                secondary_y=False,
            )
            
            if 'order_value_usd' in self.df.columns:
                fig_trend.add_trace(
                    go.Bar(
                        x=monthly_data['Month'],
                        y=monthly_data['Revenue'],
                        name='Revenue',
                        marker_color=self.colors['success'],
                        opacity=0.7
                    ),
                    secondary_y=True,
                )
            
            fig_trend.update_xaxes(title_text="Month")
            fig_trend.update_yaxes(title_text="Number of Orders", secondary_y=False)
            fig_trend.update_yaxes(title_text="Revenue ($)", secondary_y=True)
            
            fig_trend.update_layout(
                title_text="Business Performance Trend",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family="Segoe UI",
                hovermode='x unified'
            )
            st.plotly_chart(fig_trend, use_container_width=True)
    
    def _create_fallback_visualization(self, query):
        """Enhanced fallback visualization when specific ones fail"""
        st.subheader("Data Overview Dashboard")
        st.info("Generating enhanced analytics for your query...")
        
        # Enhanced metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", self.format_large_number(len(self.df)))
        with col2:
            st.metric("Data Columns", len(self.df.columns))
        
        # Enhanced sample data view
        st.subheader("Sample Data Preview")
        
        # Show data with better formatting
        sample_data = self.df.head(10)
        
        # Format currency columns
        for col in sample_data.columns:
            if 'usd' in col.lower() or 'price' in col.lower() or 'cost' in col.lower():
                sample_data[col] = sample_data[col].apply(lambda x: self.format_currency(x) if pd.notnull(x) else '')
        
        st.dataframe(sample_data, use_container_width=True)
    
    def create_comprehensive_dashboard(self):
        """Create enhanced comprehensive analytics dashboard"""
        st.header("Executive Analytics Dashboard")
        
        # Enhanced Overall KPIs
        st.subheader("Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_orders = self.df['order_id'].nunique() if 'order_id' in self.df.columns else len(self.df)
            st.metric("Total Orders", self.format_large_number(total_orders))
        
        with col2:
            if 'order_value_usd' in self.df.columns:
                total_revenue = self.df[self.df['order_status']=='Completed']['order_value_usd'].sum()
                st.metric("Total Revenue", self.format_currency(total_revenue))
        
        with col3:
            if 'is_delayed' in self.df.columns:
                delayed_count = self.df['is_delayed'].sum()
                delay_rate = (delayed_count / total_orders) * 100
                st.metric("Delay Rate", f"{delay_rate:.1f}%")
        
        with col4:
            if 'customer_satisfaction' in self.df.columns:
                avg_satisfaction = self.df['customer_satisfaction'].mean()
                st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/5")
        
        # Enhanced Revenue Analysis
        st.subheader("Revenue Analysis")
        
        if 'order_value_usd' in self.df.columns and 'order_date' in self.df.columns:
            # Monthly revenue trend with formatting
            monthly_revenue = self.df.groupby(self.df['order_date'].dt.to_period('M'))['order_value_usd'].sum()
            fig_revenue_trend = px.line(
                x=monthly_revenue.index.astype(str),
                y=monthly_revenue.values,
                title="Monthly Revenue Trend",
                labels={'x': 'Month', 'y': 'Revenue (USD)'}
            )
            
            # Enhanced hover template with formatted values
            fig_revenue_trend.update_traces(
                hovertemplate='<b>%{x}</b><br>Revenue: ' + 
                '<b>' + ','.join([self.format_currency(val) for val in monthly_revenue.values]) + '</b>'
            )
            st.plotly_chart(fig_revenue_trend, use_container_width=True)
            
            # Revenue by customer type with enhanced formatting
            if 'customer_type' in self.df.columns:
                revenue_by_type = self.df.groupby('customer_type')['order_value_usd'].sum().sort_values(ascending=False)
                fig_customer_revenue = px.bar(
                    x=revenue_by_type.index,
                    y=revenue_by_type.values,
                    title="Revenue by Customer Type",
                    labels={'x': 'Customer Type', 'y': 'Revenue (USD)'}
                )
                
                # Update hover template with formatted currency
                fig_customer_revenue.update_traces(
                    texttemplate=[self.format_currency(x) for x in revenue_by_type.values],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Revenue: <b>%{text}</b>'
                )
                st.plotly_chart(fig_customer_revenue, use_container_width=True)
        
        # Enhanced Operational Analysis
        st.subheader("Operational Analysis")
        
        if 'is_delayed' in self.df.columns:
            # Delay analysis with enhanced visuals
            delayed_orders = self.df[self.df['is_delayed'] == True]
            
            if len(delayed_orders) > 0 and 'primary_delay_category' in self.df.columns:
                delay_categories = delayed_orders['primary_delay_category'].value_counts()
                fig_delays = px.pie(
                    values=delay_categories.values,
                    names=delay_categories.index,
                    title="Delay Categories Distribution"
                )
                
                # Enhanced labels with count and percentage
                fig_delays.update_traces(
                    texttemplate='%{label}<br>%{value} orders<br>(%{percent})',
                    textposition='inside'
                )
                st.plotly_chart(fig_delays, use_container_width=True)
        
        # Enhanced Customer Analysis
        st.subheader("Customer Analysis")
        
        if 'customer_tier' in self.df.columns:
            tier_distribution = self.df['customer_tier'].value_counts()
            fig_tiers = px.bar(
                x=tier_distribution.index,
                y=tier_distribution.values,
                title="Customer Tier Distribution",
                labels={'x': 'Customer Tier', 'y': 'Number of Customers'}
            )
            
            # Enhanced formatting with text labels
            fig_tiers.update_traces(
                texttemplate=[self.format_large_number(x) for x in tier_distribution.values],
                textposition='outside'
            )
            st.plotly_chart(fig_tiers, use_container_width=True)
        
        # Enhanced Performance Metrics by Month
        st.subheader("Monthly Performance Metrics")
        
        if 'order_date' in self.df.columns:
            # Create monthly summary with enhanced formatting
            monthly_stats = self.df.groupby(self.df['order_date'].dt.to_period('M')).agg({
                'order_id': 'nunique',
                'order_value_usd': 'sum' if 'order_value_usd' in self.df.columns else 'count',
                'is_delayed': 'sum' if 'is_delayed' in self.df.columns else 'count'
            }).fillna(0)
            
            if 'order_value_usd' in self.df.columns:
                monthly_stats.columns = ['Orders', 'Revenue', 'Delayed_Orders']
                
                # Create enhanced subplot with proper formatting
                fig_monthly = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Monthly Orders', 'Monthly Revenue', 'Delayed Orders', 'Delay Rate %'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Monthly Orders with enhanced styling
                fig_monthly.add_trace(
                    go.Bar(
                        x=monthly_stats.index.astype(str), 
                        y=monthly_stats['Orders'], 
                        name='Orders', 
                        marker_color='#2E86AB',
                        text=[self.format_large_number(x) for x in monthly_stats['Orders']],
                        textposition='outside'
                    ),
                    row=1, col=1
                )
                
                # Monthly Revenue with enhanced formatting
                fig_monthly.add_trace(
                    go.Bar(
                        x=monthly_stats.index.astype(str), 
                        y=monthly_stats['Revenue'], 
                        name='Revenue', 
                        marker_color='#27AE60',
                        text=[self.format_currency(x) for x in monthly_stats['Revenue']],
                        textposition='outside'
                    ),
                    row=1, col=2
                )
                
                # Delayed Orders
                fig_monthly.add_trace(
                    go.Bar(
                        x=monthly_stats.index.astype(str), 
                        y=monthly_stats['Delayed_Orders'], 
                        name='Delayed', 
                        marker_color='#E74C3C',
                        text=[self.format_large_number(x) for x in monthly_stats['Delayed_Orders']],
                        textposition='outside'
                    ),
                    row=2, col=1
                )
                
                # Delay Rate with enhanced line
                delay_rate = (monthly_stats['Delayed_Orders'] / monthly_stats['Orders'] * 100).fillna(0)
                fig_monthly.add_trace(
                    go.Scatter(
                        x=monthly_stats.index.astype(str), 
                        y=delay_rate, 
                        mode='lines+markers', 
                        name='Delay Rate %', 
                        line=dict(color='#F39C12', width=3),
                        marker=dict(size=8),
                        text=[f"{x:.1f}%" for x in delay_rate],
                        textposition='top center'
                    ),
                    row=2, col=2
                )
                
                fig_monthly.update_layout(
                    title_text="Monthly Performance Dashboard",
                    showlegend=False,
                    height=600,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_family="Segoe UI"
                )
                
                # Update y-axis labels with proper formatting
                fig_monthly.update_yaxes(title_text="Orders", row=1, col=1)
                fig_monthly.update_yaxes(title_text="Revenue (USD)", row=1, col=2)
                fig_monthly.update_yaxes(title_text="Delayed Orders", row=2, col=1)
                fig_monthly.update_yaxes(title_text="Delay Rate (%)", row=2, col=2)
                
                st.plotly_chart(fig_monthly, use_container_width=True)
        
        # # Enhanced Data Quality Section
        # st.subheader("Data Quality Overview")
        
        # missing_data = self.df.isnull().sum()
        # missing_pct = (missing_data / len(self.df)) * 100
        # quality_df = pd.DataFrame({
        #     'Column': missing_data.index,
        #     'Missing Count': missing_data.values,
        #     'Missing %': missing_pct.values
        # }).sort_values('Missing %', ascending=False).head(10)
        
        # # Format the missing count column with enhanced formatting
        # quality_df['Missing Count Formatted'] = quality_df['Missing Count'].apply(self.format_large_number)
        
        # # Display with enhanced formatting
        # st.dataframe(
        #     quality_df[['Column', 'Missing Count Formatted', 'Missing %']].rename(columns={
        #         'Missing Count Formatted': 'Missing Count'
        #     }), 
        #     use_container_width=True
        # )
        
        # Enhanced Summary Statistics
        st.subheader("Summary Statistics")
        
        if 'order_value_usd' in self.df.columns:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                min_order = self.df['order_value_usd'].min()
                st.metric("Min Order Value", self.format_currency(min_order))
            
            with col2:
                max_order = self.df['order_value_usd'].max()
                st.metric("Max Order Value", self.format_currency(max_order))
            
            with col3:
                median_order = self.df['order_value_usd'].median()
                st.metric("Median Order Value", self.format_currency(median_order))
            
            with col4:
                std_order = self.df['order_value_usd'].std()
                st.metric("Std Deviation", self.format_currency(std_order))
        
        # Enhanced Recent Activity (Last 30 days)
        if 'order_date' in self.df.columns:
            st.subheader("Recent Activity (Last 30 Days)")
            
            last_30_days = datetime.now() - timedelta(days=30)
            recent_orders = self.df[self.df['order_date'] >= last_30_days]
            
            if len(recent_orders) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    recent_count = recent_orders['order_id'].nunique() if 'order_id' in self.df.columns else len(recent_orders)
                    st.metric("Recent Orders", self.format_large_number(recent_count))
                
                with col2:
                    if 'order_value_usd' in self.df.columns:
                        recent_revenue = recent_orders['order_value_usd'].sum()
                        st.metric("Recent Revenue", self.format_currency(recent_revenue))
                
                with col3:
                    if 'is_delayed' in self.df.columns:
                        recent_delayed = recent_orders['is_delayed'].sum()
                        recent_delay_rate = (recent_delayed / recent_count * 100) if recent_count > 0 else 0
                        st.metric("Recent Delay Rate", f"{recent_delay_rate:.1f}%")
            else:
                st.info("No orders found in the last 30 days")
        
        # Enhanced Top Customers by Revenue
        if 'customer_name' in self.df.columns and 'order_value_usd' in self.df.columns:
            st.subheader("Top Customers by Revenue")
            
            top_customers = self.df.groupby('customer_name')['order_value_usd'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(10)
            top_customers.columns = ['Total Revenue', 'Order Count']
            
            # Enhanced formatting for the revenue column
            top_customers['Total Revenue Formatted'] = top_customers['Total Revenue'].apply(self.format_currency)
            top_customers['Order Count Formatted'] = top_customers['Order Count'].apply(self.format_large_number)
            
            st.dataframe(
                top_customers[['Total Revenue Formatted', 'Order Count Formatted']].rename(columns={
                    'Total Revenue Formatted': 'Total Revenue',
                    'Order Count Formatted': 'Order Count'
                }), 
                use_container_width=True
            )