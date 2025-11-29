"""
ShapeNetPart label mappings.

Based on the ShapeNetPart dataset which has 16 object categories
and 50 part classes total across all categories.

Reference: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""

from typing import Optional, Tuple

# ShapeNetPart category to part label mapping
# Each category has a set of part labels (0-indexed within that category)
# The actual model uses a flat label space (0-49) across all categories
SHAPENETPART_CATEGORY_LABELS = {
    "Airplane": {
        0: "body",
        1: "engine",
        2: "wing",
        3: "tail",
        4: "propeller",
    },
    "Bag": {
        0: "body",
        1: "handle",
        2: "wheel",
    },
    "Cap": {
        0: "brim",
        1: "crown",
    },
    "Car": {
        0: "body",
        1: "wheel",
        2: "door",
        3: "hood",
        4: "mirror",
        5: "light",
        6: "license_plate",
        7: "roof",
    },
    "Chair": {
        0: "back",
        1: "seat",
        2: "leg",
        3: "armrest",
    },
    "Earphone": {
        0: "earcup",
        1: "headband",
        2: "wire",
    },
    "Guitar": {
        0: "body",
        1: "neck",
        2: "head",
        3: "string",
    },
    "Knife": {
        0: "blade",
        1: "handle",
    },
    "Lamp": {
        0: "base",
        1: "shade",
        2: "pole",
        3: "bulb",
    },
    "Laptop": {
        0: "screen",
        1: "keyboard",
        2: "touchpad",
        3: "base",
    },
    "Motorbike": {
        0: "wheel",
        1: "handlebar",
        2: "saddle",
        3: "body",
        4: "headlight",
    },
    "Mug": {
        0: "body",
        1: "handle",
    },
    "Pistol": {
        0: "barrel",
        1: "handle",
        2: "trigger",
    },
    "Rocket": {
        0: "body",
        1: "fin",
        2: "nose",
    },
    "Skateboard": {
        0: "board",
        1: "wheel",
    },
    "Table": {
        0: "top",
        1: "leg",
    },
}

# Flat label space mapping (0-49) used by the model
# This is a simplified mapping - actual mapping depends on training data
# In practice, you may need to adjust based on your pretrained model
FLAT_LABEL_TO_CATEGORY_PART = {}
category_offset = 0
for category, parts in SHAPENETPART_CATEGORY_LABELS.items():
    for local_id, part_name in parts.items():
        flat_id = category_offset + local_id
        FLAT_LABEL_TO_CATEGORY_PART[flat_id] = (category, part_name)
    category_offset += len(parts)

# Reverse mapping: category + part_name -> flat label
CATEGORY_PART_TO_FLAT_LABEL = {}
flat_id = 0
for category, parts in SHAPENETPART_CATEGORY_LABELS.items():
    for local_id, part_name in parts.items():
        CATEGORY_PART_TO_FLAT_LABEL[(category, part_name)] = flat_id
        flat_id += 1


def get_label_name(category: str, label_id: int) -> Optional[str]:
    """
    Get part name for a label ID within a category.
    
    Args:
        category: object category (e.g., "Airplane")
        label_id: local label ID within category (0-indexed)
        
    Returns:
        Part name or None if not found
    """
    if category not in SHAPENETPART_CATEGORY_LABELS:
        return None
    return SHAPENETPART_CATEGORY_LABELS[category].get(label_id)


def get_flat_label(category: str, part_name: str) -> Optional[int]:
    """
    Get flat label ID (0-49) for a category + part name.
    
    Args:
        category: object category
        part_name: part name
        
    Returns:
        Flat label ID or None if not found
    """
    return CATEGORY_PART_TO_FLAT_LABEL.get((category, part_name))


def get_category_from_flat_label(flat_label: int) -> Optional[Tuple[str, str]]:
    """
    Get category and part name from flat label ID.
    
    Args:
        flat_label: flat label ID (0-49)
        
    Returns:
        Tuple (category, part_name) or None if not found
    """
    return FLAT_LABEL_TO_CATEGORY_PART.get(flat_label)


# List of all categories
SHAPENETPART_CATEGORIES = list(SHAPENETPART_CATEGORY_LABELS.keys())

# Total number of part classes (50)
NUM_PART_CLASSES = 50

