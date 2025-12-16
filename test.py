# import matplotlib.pyplot as plt
# import numpy as np

# def generate_llm_growth_graph(filename="llm_growth.png"):
#     """
#     Generates a line graph showing the exponential growth in LLM parameter count.
#     """
#     # Approximate data for LLM parameter growth (illustrative, not precise historical data)
#     # Year and corresponding parameter count (in billions)
#     years = np.array([2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
#     # Values are highly approximate for illustration of exponential growth
#     # Real values: BERT (0.3B), GPT-2 (1.5B), GPT-3 (175B), PaLM (540B), Gemini (likely vastly more)
#     # Using a simplified exponential scale for visual impact
#     parameter_counts = np.array([0.1, 1.5, 175, 500, 1000, 2000, 4000, 8000]) # Example scale in billions

#     plt.figure(figsize=(10, 6))
#     plt.plot(years, parameter_counts, marker='o', linestyle='-', color='skyblue', linewidth=2, markersize=8)

#     plt.title('Exponential Growth in Large Language Model Parameter Count', fontsize=16)
#     plt.xlabel('Year', fontsize=12)
#     plt.ylabel('Parameter Count (Billions)', fontsize=12)
#     plt.yscale('log') # Use a logarithmic scale for y-axis to better show exponential growth
#     plt.grid(True, which="both", ls="--", c='0.7')
#     plt.xticks(years, fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.tick_params(axis='x', rotation=45)

#     plt.annotate('GPT-3 (175B)', xy=(2020, 175), xytext=(2020.5, 250),
#                  arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
#                  fontsize=9, color='darkblue')
#     plt.annotate('PaLM (540B)', xy=(2021, 500), xytext=(2021.5, 700),
#                  arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
#                  fontsize=9, color='darkblue')

#     plt.tight_layout()
#     plt.savefig(filename, dpi=300)
#     plt.close()
#     print(f"Generated {filename}")

# if __name__ == "__main__":
#     generate_llm_growth_graph()


# import matplotlib.pyplot as plt

# def generate_governance_comparison_table(filename="governance_comparison.png"):
#     """
#     Generates a table comparing major AI governance approaches.
#     """
#     data = [
#         ["European Union", "Rights-based", "Legislation (EU AI Act)", "Comprehensive, binding, risk-pyramid"],
#         ["United States", "Market-driven", "Executive Orders, Agency Guidelines", "Voluntary, promotes innovation"],
#         ["China", "State-centric", "Mandates, Regulations", "Content moderation, state control"],
#     ]

#     columns = ("Region", "Focus", "Mechanism", "Key Characteristics")
#     rows = [row[0] for row in data]

#     fig, ax = plt.subplots(figsize=(12, 4)) # Adjust figure size as needed
#     ax.axis('tight')
#     ax.axis('off')

#     # Create the table
#     table = ax.table(cellText=data,
#                      colLabels=columns,
#                      loc='center',
#                      cellLoc='center',
#                      colColours=["#f2f2f2"] * len(columns)) # Light grey background for column headers

#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1.2, 1.2) # Scale cell width/height

#     # Customizing cell appearance
#     for (i, j), cell in table.get_celld().items():
#         cell.set_edgecolor('black')
#         cell.set_linewidth(0.5)
#         if i == 0: # Header row
#             cell.set_facecolor('#d9d9d9') # Darker grey for headers
#             cell.set_text_props(weight='bold')
#         else:
#             cell.set_facecolor('white')

#     plt.title('Comparison of Major AI Governance Approaches', fontsize=14, pad=20)
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300)
#     plt.close()
#     print(f"Generated {filename}")

# if __name__ == "__main__":
#     generate_governance_comparison_table()

# ### 3. `alignment_problem.png` (AI Alignment Conceptual Illustration - Simplified)

# # This script attempts to create a *very basic* conceptual diagram. For anything more detailed or visually sophisticated, a dedicated diagramming tool is highly recommended.

# # ```python
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# def generate_alignment_problem_diagram(filename="alignment_problem.png"):
#     """
#     Generates a simplified conceptual illustration of the AI Alignment Problem.
#     NOTE: For complex diagrams, dedicated tools (draw.io, Lucidchart) are better.
#     """
#     fig, ax = plt.subplots(figsize=(10, 7))
#     ax.set_xlim(0, 10)
#     ax.set_ylim(0, 10)
#     ax.axis('off') # Hide axes

#     # Define nodes (rectangles)
#     node_width = 2.5
#     node_height = 1.2
#     font_size = 10

#     # Node 1: Human Values
#     human_values_rect = patches.Rectangle((1, 8), node_width, node_height,
#                                           linewidth=1.5, edgecolor='black', facecolor='#DDEBF7', zorder=2)
#     ax.add_patch(human_values_rect)
#     ax.text(1 + node_width/2, 8 + node_height/2, 'Human Values\n(What we want)',
#             ha='center', va='center', fontsize=font_size, weight='bold', zorder=3)

#     # Node 2: AI Goal Specification (Outer Alignment)
#     ai_spec_rect = patches.Rectangle((4.5, 8), node_width, node_height,
#                                      linewidth=1.5, edgecolor='black', facecolor='#FBE4D5', zorder=2)
#     ax.add_patch(ai_spec_rect)
#     ax.text(4.5 + node_width/2, 8 + node_height/2, 'AI Goal Specification\n(Outer Alignment)',
#             ha='center', va='center', fontsize=font_size, weight='bold', zorder=3)

#     # Node 3: AI Model (Inner Alignment)
#     ai_model_rect = patches.Rectangle((3, 3), node_width + 3, node_height + 2,
#                                       linewidth=1.5, edgecolor='black', facecolor='#E2EFDA', zorder=2)
#     ax.add_patch(ai_model_rect)
#     ax.text(3 + (node_width + 3)/2, 3 + (node_height + 2)/2, 'AI Model\n(Inner Alignment: Robust Goals, No Deception)',
#             ha='center', va='center', fontsize=font_size, weight='bold', zorder=3)

#     # Arrows and Labels
#     # Human Values -> AI Goal Specification
#     ax.annotate("", xy=(1 + node_width, 8.6), xytext=(4.5, 8.6),
#                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=1.5, color='darkgreen'))
#     ax.text(3.75, 8.8, 'Specifying Goals', ha='center', va='bottom', fontsize=font_size-1, color='darkgreen')

#     # AI Goal Specification -> AI Model
#     ax.annotate("", xy=(5.75, 8), xytext=(5.75, 5.2 + node_height/2),
#                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=1.5, color='darkblue'))
#     ax.text(6.05, 6.7, 'Training', ha='left', va='center', fontsize=font_size-1, color='darkblue')

#     # Feedback from AI Model
#     ax.annotate("", xy=(6.5 + node_width/2, 3 + node_height), xytext=(6.5 + node_width/2, 7.5),
#                 arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0", lw=1.5, color='purple'))
#     ax.text(6.75, 6.7, 'Feedback / Evaluations', ha='left', va='center', fontsize=font_size-1, color='purple')

#     # Title
#     plt.title('Conceptual Illustration of the AI Alignment Problem', fontsize=16, pad=20)

#     plt.tight_layout()
#     plt.savefig(filename, dpi=300)
#     plt.close()
#     print(f"Generated {filename}")

# if __name__ == "__main__":
#     generate_alignment_problem_diagram()

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_detailed_alignment_problem_diagram(filename="alignment_problem.png"):
    """
    Generates a conceptual illustration of the AI Alignment Problem,
    incorporating details and citations from the provided description.
    """
    fig, ax = plt.subplots(figsize=(12, 8)) # Increased size for more text
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off') # Hide axes

    # Define common properties for nodes (rectangles)
    node_width = 2.8
    node_height_top = 1.3
    node_height_bottom = 2.5 # Taller for the AI Model box
    font_size_main = 10
    font_size_label = 9

    # Node 1: Human Values
    human_values_rect = patches.Rectangle((0.8, 7.8), node_width, node_height_top,
                                          linewidth=1.5, edgecolor='black', facecolor='#DDEBF7', zorder=2)
    ax.add_patch(human_values_rect)
    ax.text(0.8 + node_width/2, 7.8 + node_height_top/2,
            'Human Values\n(What we want) [3,4,5]',
            ha='center', va='center', fontsize=font_size_main, weight='bold', wrap=True, zorder=3)

    # Node 2: AI Goal Specification (Outer Alignment)
    ai_spec_rect = patches.Rectangle((4.5, 7.8), node_width, node_height_top,
                                     linewidth=1.5, edgecolor='black', facecolor='#FBE4D5', zorder=2)
    ax.add_patch(ai_spec_rect)
    ax.text(4.5 + node_width/2, 7.8 + node_height_top/2,
            'AI Goal Specification\n(Outer Alignment) [6,7,8,9]',
            ha='center', va='center', fontsize=font_size_main, weight='bold', wrap=True, zorder=3)
    # Adding a note about reward hacking/specification gaming near outer alignment
    ax.text(4.5 + node_width/2, 7.8 - 0.3,
            'Failures can lead to "reward hacking" or\n"specification gaming" [1,2,4,6]',
            ha='center', va='top', fontsize=font_size_label-1, color='gray', wrap=True)


    # Node 3: AI Model (Inner Alignment)
    ai_model_rect = patches.Rectangle((3, 2.5), node_width + 3, node_height_bottom,
                                      linewidth=1.5, edgecolor='black', facecolor='#E2EFDA', zorder=2)
    ax.add_patch(ai_model_rect)
    ax.text(3 + (node_width + 3)/2, 2.5 + node_height_bottom/2,
            'AI Model\n(Inner Alignment: Robust Goals, No Deception) [6,7,9,10,11,12,13,14,15]',
            ha='center', va='center', fontsize=font_size_main, weight='bold', wrap=True, zorder=3)
    # Adding a note about instrumental goals/unpredictability
    ax.text(3 + (node_width + 3)/2, 2.5 - 0.3,
            'Risk of hidden, instrumental goals & unpredictable behavior [6,9,16]',
            ha='center', va='top', fontsize=font_size_label-1, color='gray', wrap=True)


    # Arrows and Labels
    # Human Values -> AI Goal Specification (Specifying Goals)
    ax.annotate("", xy=(0.8 + node_width, 7.8 + node_height_top/2), xytext=(4.5, 7.8 + node_height_top/2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=1.5, color='darkgreen'))
    ax.text(3.65, 7.8 + node_height_top/2 + 0.1, 'Specifying Goals', ha='center', va='bottom',
            fontsize=font_size_label, color='darkgreen')

    # AI Goal Specification -> AI Model (Training)
    ax.annotate("", xy=(4.5 + node_width/2, 7.8), xytext=(4.5 + node_width/2, 2.5 + node_height_bottom),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=1.5, color='darkblue'))
    ax.text(4.5 + node_width/2 + 0.1, 5.5, 'Training', ha='left', va='center',
            fontsize=font_size_label, color='darkblue', rotation=90)

    # AI Model -> AI Goal Specification (Feedback / Evaluations)
    ax.annotate("", xy=(3 + (node_width + 3)/2 + 0.5, 2.5 + node_height_bottom), xytext=(3 + (node_width + 3)/2 + 0.5, 7.8),
                arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0", lw=1.5, color='purple'))
    ax.text(3 + (node_width + 3)/2 + 0.5 + 0.1, 5.5, 'Feedback / Evaluations', ha='right', va='center',
            fontsize=font_size_label, color='purple', rotation=90)


    # Overall Title
    plt.title('Conceptual Illustration of the AI Alignment Problem', fontsize=16, pad=20)

    # Note about existential risk (placed at bottom)
    ax.text(5, 0.5,
            'This diagram is central to understanding existential risks from AI and the\n'
            'potential for misaligned superintelligence [3,17,18,19,20].',
            ha='center', va='center', fontsize=font_size_label, color='black', wrap=True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Generated {filename}")

if __name__ == "__main__":
    generate_detailed_alignment_problem_diagram()