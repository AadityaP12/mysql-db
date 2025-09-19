import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Load the cleaned dataset
    df = pd.read_csv('C:/SIH/WQuality_River_Data_2023_Cleaned.csv')

    # --- 1. Identify Top 5 States with the Most Monitoring Stations ---
    top_states = df['State'].value_counts().nlargest(5)
    print("--- Top 5 States with Most Monitoring Stations ---")
    print(top_states)

    # Filter the DataFrame to include only the top 5 states
    top_states_df = df[df['State'].isin(top_states.index)]

    # --- 2. Calculate pH Statistics for the Top States ---
    # We will use the average of pH_Min and pH_Max to get a representative pH value
    top_states_df['pH_Avg'] = top_states_df[['pH_Min', 'pH_Max']].mean(axis=1)

    # Group by state and calculate the required statistics
    ph_stats = top_states_df.groupby('State')['pH_Avg'].agg(['mean', 'min', 'max']).round(2)
    print("\n--- pH Statistics for Top 5 States ---")
    print(ph_stats)

    # --- 3. Visualize pH Distribution with a Box Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create the box plot
    sns.boxplot(x='State', y='pH_Avg', data=top_states_df, order=top_states.index, palette='viridis')

    # Add titles and labels for clarity
    ax.set_title('pH Level Distribution in Top 5 Monitored States', fontsize=16, weight='bold')
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Average pH', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('C:/SIH/pH_Distribution_Top_5_States.png')

    print("\nAnalysis complete! Box plot saved to 'pH_Distribution_Top_5_States.png'")

except FileNotFoundError:
    print("Error: 'WQuality_River_Data_2023_Cleaned.csv' not found.")
    print("Please make sure the cleaned CSV file is in the same directory as this script.")