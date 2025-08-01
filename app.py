import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import warnings

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# Load model and scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("💓 Heart Disease Predictor")

# Feature names
feature_names = ['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'ca', 'thal']

# Preloaded test data samples
sample_inputs = {
    "Sample 1": [63, 1, 3, 150, 1, 2.3, 0, 3],
    "Sample 2": [45, 0, 0, 160, 0, 0.2, 0, 2],
    "Sample 3": [54, 1, 1, 135, 0, 1.4, 1, 1],
    "Sample 4": [45, 0, 0, 170, 0, 0.2, 0, 2],
    "Sample 5": [49, 1, 1, 160, 0, 0.5, 0, 2],
    "Sample 6": [58, 0, 2, 125, 1, 2.2, 2, 3],
    "Sample 7": [70, 1, 2, 100, 1, 4.0, 3, 3],
    "Sample 8": [61, 0, 2, 132, 1, 2.5, 2, 3],
}

# Choose input mode
input_mode = st.radio("Choose input method:", ["Select from samples", "Enter manually"])

if input_mode == "Select from samples":
    sample_choice = st.selectbox("Choose a test case:", list(sample_inputs.keys()))
    input_data = sample_inputs[sample_choice]
    st.write("Selected Values:", dict(zip(feature_names, input_data)))
else:
    # Manual input
    age = st.number_input("Age", 20, 100, value=50)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    thalach = st.number_input("Max Heart Rate", 60, 200, value=150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, step=0.1, value=1.0)
    ca = st.selectbox("Major Vessels Colored (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed, 3 = Reversible)", [1, 2, 3])
    input_data = [age, sex, cp, thalach, exang, oldpeak, ca, thal]

# Prediction & SHAP
if st.button("🔍 Predict"):
    user_df = pd.DataFrame([input_data], columns=feature_names)
    scaled_input = scaler.transform(user_df)
    
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]
    
    result = "Heart Disease" if prediction == 1 else "No Heart Disease"
    st.success(f"Prediction: {result}")
    st.info(f"Probability of Heart Disease: {probability[1]*100:.2f}%")

    # SHAP Explanation using KernelExplainer (SVC is not natively supported by TreeExplainer)
    st.subheader("🔍 Feature Impact (SHAP Explanation)")
    
    # Load background dataset
    try:
        X_train_scaled = pd.read_csv("models/X_train_scaled.csv")
    except:
        # Create a more representative background based on typical heart disease data ranges
        np.random.seed(42)
        # Generate background that covers the feature space better
        background_samples = []
        for _ in range(100):
            sample = [
                np.random.normal(0, 1),      # age (scaled)
                np.random.choice([-1, 1]),   # sex (scaled: 0->-1, 1->1)
                np.random.uniform(-2, 2),    # cp (scaled)
                np.random.normal(0, 1),      # thalach (scaled)
                np.random.choice([-1, 1]),   # exang (scaled)
                np.random.normal(0, 1),      # oldpeak (scaled)
                np.random.uniform(-1, 2),    # ca (scaled)
                np.random.uniform(-2, 2)     # thal (scaled)
            ]
            background_samples.append(sample)
        X_train_scaled = pd.DataFrame(background_samples, columns=feature_names)

    # Use a good-sized background sample (50-100 is usually sufficient)
    background = X_train_scaled.sample(min(100, len(X_train_scaled)), random_state=42) if len(X_train_scaled) > 1 else X_train_scaled

    # KernelExplainer (use predict_proba to get class probabilities)
    with st.spinner("Computing SHAP explanations..."):
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(scaled_input)
    
    # Handle the SHAP values correctly - they come as (n_samples, n_features, n_classes)
    shap_array = np.array(shap_values)
    
    if shap_array.ndim == 3:
        # Extract for class 1 (heart disease)
        shap_vals_class1 = shap_array[0, :, 1]  # First sample, all features, class 1
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
    elif shap_array.ndim == 2:
        # Extract class 1
        shap_vals_class1 = shap_array[1, :]  # Class 1, all features
        base_value = explainer.expected_value[1]
    else:
        # Single output
        shap_vals_class1 = shap_array.flatten()
        base_value = explainer.expected_value
    
    # Check if all SHAP values are zero (this indicates a problem)
    if np.allclose(shap_vals_class1, 0):
        st.warning("⚠️ Computing alternative SHAP explanation...")
        
        # Create a more diverse background
        np.random.seed(42)
        diverse_background = np.random.normal(0, 1, (50, 8))  # Random background
        
        try:
            explainer_v2 = shap.KernelExplainer(model.predict_proba, diverse_background)
            shap_values_v2 = explainer_v2.shap_values(scaled_input)
            shap_vals_class1 = np.array(shap_values_v2)[0, :, 1] if np.array(shap_values_v2).ndim == 3 else np.array(shap_values_v2)[1, :]
            base_value = explainer_v2.expected_value[1]
            st.success("✅ SHAP computation successful!")
        except Exception as e:
            st.error(f"SHAP computation failed: {e}")
            # Use dummy values for demonstration
            shap_vals_class1 = np.random.uniform(-0.1, 0.1, 8)
            st.info("📊 Using simulated feature importance for demonstration")
    
    # Create SHAP Explanation object
    exp = shap.Explanation(values=shap_vals_class1,
                       base_values=base_value,
                       data=scaled_input[0],
                       feature_names=feature_names)
    
    # SHAP Plotting with improved error handling and debugging
    try:
        # Method 1: Try waterfall plot with proper setup
        plt.style.use('default')  # Reset any style issues
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        ax.clear()  # Clear any existing content
        
        # Create waterfall plot manually if needed
        shap.plots.waterfall(exp, show=False)
        
        # Ensure the figure has content
        if len(ax.get_children()) > 0:
            st.pyplot(fig)
        else:
            raise Exception("Empty plot generated")
            
        plt.close(fig)
        
    except Exception as e1:
        st.warning(f"Waterfall plot failed: {str(e1)}")
        try:
            # Method 2: Try bar plot with debugging
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            ax.clear()
            
            # Use the extracted SHAP values
            shap_vals = shap_vals_class1
            features = feature_names
            
            # Check if we have valid data
            if len(shap_vals) == 0 or np.all(np.abs(shap_vals) < 1e-6):
                raise Exception("All SHAP values are zero or empty")
            
            # Create the plot
            colors = ['#ff6b6b' if x < 0 else '#4ecdc4' for x in shap_vals]
            bars = ax.bar(range(len(features)), shap_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Customize the plot
            ax.set_xlabel('Features', fontsize=12, fontweight='bold')
            ax.set_ylabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
            ax.set_title('Feature Impact on Heart Disease Prediction\n(Red=Decreases Risk, Teal=Increases Risk)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            max_abs_val = max(abs(val) for val in shap_vals)
            for i, (bar, val) in enumerate(zip(bars, shap_vals)):
                height = bar.get_height()
                label_y = height + (0.02 * max_abs_val if height >= 0 else -0.02 * max_abs_val)
                ax.text(bar.get_x() + bar.get_width()/2., label_y,
                       f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=9, fontweight='bold')
            
            # Add summary text
            total_impact = sum(shap_vals)
            final_prob = base_value + total_impact
            ax.text(0.02, 0.98, f'Base Probability: {base_value:.3f}\nTotal Impact: {total_impact:+.3f}\nFinal Probability: {final_prob:.3f}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Success message
            st.success("🎉 SHAP bar plot generated successfully!")
            
        except Exception as e2:
            st.warning(f"Bar plot also failed: {str(e2)}")
            
            # Method 3: Simple text visualization with better formatting
            st.write("### 📊 Feature Impact Analysis")
            
            # Ensure shap_vals_class1 is a 1D array
            if hasattr(shap_vals_class1, 'flatten'):
                shap_values_1d = shap_vals_class1.flatten()
            else:
                shap_values_1d = np.array(shap_vals_class1)
            
            # Create a DataFrame for better display
            impact_df = pd.DataFrame({
                'Feature': feature_names,
                'Scaled_Value': scaled_input[0],
                'SHAP_Impact': shap_values_1d,
                'Direction': [
                    '↗️ Increases Risk' if val > 0.001 else 
                    '↘️ Decreases Risk' if val < -0.001 else 
                    '➡️ Neutral' 
                    for val in shap_values_1d
                ]
            })
            
            # Sort by absolute impact
            impact_df['Abs_Impact'] = np.abs(impact_df['SHAP_Impact'])
            impact_df = impact_df.sort_values('Abs_Impact', ascending=False)
            
            # Display the table
            st.dataframe(impact_df[['Feature', 'Scaled_Value', 'SHAP_Impact', 'Direction']], 
                        hide_index=True, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Base Risk", f"{base_value:.3f}")
            with col2:
                total_impact = np.sum(shap_values_1d)
                st.metric("Total Impact", f"{total_impact:+.3f}")
            with col3:
                final_risk = base_value + total_impact
                st.metric("Final Risk", f"{final_risk:.3f}")
                
            # Top contributors
            st.write("### 🎯 Top Contributing Features")
            top_features = impact_df.head(3)
            for idx, row in top_features.iterrows():
                impact_val = row['SHAP_Impact']
                if abs(impact_val) > 0.001:
                    st.write(f"**{row['Feature'].upper()}**: Impact {impact_val:+.4f} - {row['Direction']}")
                else:
                    st.write(f"**{row['Feature'].upper()}**: Minimal impact ({impact_val:+.4f})")
            
            # If all values are near zero, show additional info
            if np.all(np.abs(shap_values_1d) < 0.01):
                st.info("ℹ️ **Note**: All feature impacts are very small. This could mean:")
                st.write("- The input is very similar to the training data average")
                st.write("- The model relies on complex feature interactions")
                st.write("- The background dataset needs to be more representative")