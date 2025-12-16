
import torch
import transformer_lens
from transformer_lens import HookedTransformer
import circuitsvis as cv
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
import os

def visualize_llm_internals(model_name="gpt2-small", prompt="The quick brown fox jumps over the lazy dog"):
    print(f"Loading model: {model_name}...")
    try:
        model = HookedTransformer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return

    print("Model loaded.")
    print(f"Processing prompt: '{prompt}'")

    # 1. Run model and cache activations
    logits, cache = model.run_with_cache(prompt)
    tokens = model.to_str_tokens(prompt)
    
    print("Extracting embeddings and layer activations...")
    
    # 2. Extract Token Embeddings (Input)
    # Shape: [batch, pos, d_model]
    input_embeddings = cache["embed"] 
    
    # 3. Trace a specific token's trajectory through layers
    # let's trace the last token
    token_index = -1 
    layer_trajectory = []
    
    # Add input embedding
    layer_trajectory.append(input_embeddings[0, token_index, :].detach().cpu().numpy())
    
    # Add hidden states after each layer
    for i in range(model.cfg.n_layers):
        # Result of the residual stream at layer i
        # hook name: blocks.{i}.hook_resid_post
        hidden = cache[f"blocks.{i}.hook_resid_post"][0, token_index, :].detach().cpu().numpy()
        layer_trajectory.append(hidden)
        
    layer_trajectory = np.array(layer_trajectory)
    
    # 4. Dimensionality Reduction for Visualization (PCA to 3D)
    print("Computing 3D projection of layer trajectory...")
    pca = PCA(n_components=3)
    # detailed trajectory
    traj_3d = pca.fit_transform(layer_trajectory)
    
    # 5. Create 3D Plot of Trajectory
    fig = go.Figure(data=[go.Scatter3d(
        x=traj_3d[:,0], y=traj_3d[:,1], z=traj_3d[:,2],
        mode='lines+markers+text',
        text=[f"In" if i==0 else f"L{i-1}" for i in range(len(traj_3d))],
        marker=dict(size=5, color=list(range(len(traj_3d))), colorscale='Viridis'),
        line=dict(color='darkblue', width=2)
    )])
    fig.update_layout(title=f"Token Trajectory through {model.cfg.n_layers} Layers: '{tokens[token_index]}'",
                      scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'))
    
    output_html_traj = "embedding_trajectory.html"
    fig.write_html(output_html_traj)
    print(f"Saved trajectory visualization to {output_html_traj}")

    # 6. Visualize Attention Patterns (Layer 0, all heads)
    print("Generating attention pattern visualization...")
    # Get attention pattern for layer 0: [n_heads, dest_pos, src_pos]
    attention_pat = cache["blocks.0.attn.hook_pattern"][0] 
    
    # CircuitsVis creates interactive HTML
    attn_viz = cv.attention.attention_patterns(tokens=tokens, attention=attention_pat)
    
    # Save CircuitsVis to HTML file manually since it returns a display object
    output_html_attn = "attention_patterns.html"
    with open(output_html_attn, "w") as f:
        f.write(str(attn_viz))
    print(f"Saved attention visualization to {output_html_attn}")
    
    print("\nDone! Open the generated HTML files in your browser.")

if __name__ == "__main__":
    # You can change the model name here to "gpt2-medium", "gpt2-large", "gpt2-xl" 
    # or even EleutherAI/gpt-neo-125M etc.
    # Note: Loading local GGUF models directly into TransformerLens is tricky and requires specific conversion.
    # We use gpt2-small as a proxy to demonstrate the visualization.
    visualize_llm_internals()
