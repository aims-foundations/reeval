
from huggingface_hub import HfApi
import re


def extract_parameter_count_from_name(model_name):
    """Extract parameter count from model name using patterns like xxB, xxb, xxM, xxm"""
    # Pattern to match numbers followed by 'b', 'B', 'm', or 'M'
    # Handles: 7b, 13B, 7.5b, 0.5B, 72b, 400m, 1.3M, etc.
    pattern = r'(\d+(?:\.\d+)?)[bBmM](?!it|od|el)'
    
    matches = re.findall(pattern, model_name)
    
    if matches:
        # Convert all matches to billions for comparison
        param_counts_in_billions = []
        
        for match in matches:
            # Get the unit (last character that matched)
            unit_match = re.search(rf'{re.escape(match)}([bBmM])', model_name)
            if unit_match:
                unit = unit_match.group(1).lower()
                param_value = float(match)
                
                if unit == 'b':
                    # Already in billions
                    param_counts_in_billions.append(param_value)
                elif unit == 'm':
                    # Convert millions to billions
                    param_counts_in_billions.append(param_value / 1000)
        
        if param_counts_in_billions:
            # Return the largest parameter count (usually the main model size)
            return max(param_counts_in_billions)
    
    return None


# Initialize the API client
api = HfApi()

# Get text generation models using the newer API (ModelFilter is deprecated)
# Valid sort options are: "lastModified", "downloads", "likes", "createdAt"
total_models = 300000  # Set a high limit to ensure we get enough models
model_list = list(api.list_models(
    task="text-generation",  # Direct parameter instead of ModelFilter
    sort="downloads",        # Use "downloads" instead of "trending" 
    direction=-1,           # Descending order (most downloads first)
    limit=total_models              # Get more to have options after filtering
    )
)

# Filter models with <= 30B parameters
filtered_models = []

print(f"Filtering models...{total_models}")
for model in model_list:
    param_count = extract_parameter_count_from_name(model.id)
    
    # if param_count is not None and param_count <= 30:
    if param_count and param_count <= 30:
        filtered_models.append((model.id, param_count))
    
# print(filtered_models)


print(f"\nFound {len(filtered_models)} models with ≤30B parameters out of {total_models} models.")
print("=" * 80)

