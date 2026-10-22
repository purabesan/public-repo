
import bpy
import bmesh
import numpy as np
from mathutils import Vector

# ============================================================
# Settings
# ============================================================

ALPHA_THRESHOLD = 0.05

# 3: moderate, 4: detailed, 5: high polygon count
SUBDIVISION_LEVELS = 4

# Fraction of transparent samples required for deletion.
# 1.0 is conservative. 0.75 cuts closer to the edge.
DELETE_RATIO = 0.75

# Override texture names only if auto detection fails.
# Example: {0: "Image_15"}
IMAGE_BY_SLOT = {}

# ============================================================
# Prepare source object and images
# ============================================================

src = bpy.context.active_object

if src is None or src.type != "MESH":
    raise RuntimeError("Select a Mesh object.")

if src.mode != "OBJECT":
    bpy.ops.object.mode_set(mode="OBJECT")

if src.data.shape_keys is not None:
    raise RuntimeError(
        "Shape keys detected. This script does not "
        "support shape-key interpolation safely."
    )

if not src.data.uv_layers.active:
    raise RuntimeError("Active UV map is missing.")

images = {}

def resolve_image(mat, slot_index):
    if slot_index in IMAGE_BY_SLOT:
        name = IMAGE_BY_SLOT[slot_index]
        img = bpy.data.images.get(name)
        if img is None:
            raise RuntimeError(f"Image not found: {name}")
        return img

    if not mat or not mat.use_nodes:
        raise RuntimeError(f"Invalid material: {slot_index}")

    nodes = [
        n for n in mat.node_tree.nodes
        if n.type == "TEX_IMAGE"
        and n.image is not None
        and (
            "lit color texture" in (
                n.name + " " + n.label
            ).lower()
            or "basecolortexture.image" in n.name.lower()
            or "base_color_texture.image" in n.name.lower()
        )
    ]

    if len(nodes) != 1:
        raise RuntimeError(
            f"Cannot resolve Lit Color Texture "
            f"for slot {slot_index}: {mat.name}"
        )

    return nodes[0].image

print("\n=== Alpha Boundary Cleanup ===")

for i, slot in enumerate(src.material_slots):
    img = resolve_image(slot.material, i)

    if img.source == "TILED":
        raise RuntimeError("UDIM is unsupported.")

    w, h = img.size

    if w < 1 or h < 1 or img.channels < 4:
        raise RuntimeError(
            f"Invalid RGBA image: {img.name}"
        )

    if img.name not in images:
        buf = np.empty(
            w * h * img.channels, dtype=np.float32
        )
        img.pixels.foreach_get(buf)

        images[img.name] = buf.reshape(
            h, w, img.channels
        )[:, :, 3].copy()

    print(f"Material {i}: {slot.material.name}")
    print(f"  Image: {img.name}")

# Map each material slot to its alpha pixels.
alpha_by_slot = {}

for i, slot in enumerate(src.material_slots):
    img = resolve_image(slot.material, i)
    alpha_by_slot[i] = images[img.name]

# ============================================================
# Duplicate object (preserve original)
# ============================================================

obj = src.copy()
obj.data = src.data.copy()
obj.name = src.name + "_AlphaCut"

collection = src.users_collection[0]
collection.objects.link(obj)

# Keep modifiers and vertex groups from copied object.
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

# ============================================================
# Build BMesh
# ============================================================

bm = bmesh.new()
bm.from_mesh(obj.data)

uv_layer = bm.loops.layers.uv.active

if uv_layer is None:
    bm.free()
    raise RuntimeError("BMesh UV layer missing.")

for f in bm.faces:
    f.hide_set(False)

# ============================================================
# Alpha sampling
# ============================================================

def alpha_at(alpha_pixels, uv):
    h, w = alpha_pixels.shape

    # Clamp UV to 0..1.
    u = max(0.0, min(1.0, float(uv.x)))
    v = max(0.0, min(1.0, float(uv.y)))

    x = min(w - 1, int(u * w))
    y = min(h - 1, int(v * h))

    return float(alpha_pixels[y, x])


def face_samples(face):
    alpha = alpha_by_slot.get(face.material_index)

    if alpha is None:
        return []

    uvs = [
        loop[uv_layer].uv.copy()
        for loop in face.loops
    ]

    samples = []

    # UV polygon corners.
    for uv in uvs:
        samples.append(alpha_at(alpha, uv))

    # Edge midpoints.
    for i in range(len(uvs)):
        midpoint = (
            uvs[i] + uvs[(i + 1) % len(uvs)]
        ) * 0.5
        samples.append(alpha_at(alpha, midpoint))

    # Center sample.
    center = sum(
        uvs, Vector((0.0, 0.0))
    ) / len(uvs)

    samples.append(alpha_at(alpha, center))

    return samples


def face_state(face):
    values = face_samples(face)

    if not values:
        return "KEEP", 0.0

    transparent = sum(
        a < ALPHA_THRESHOLD for a in values
    )

    ratio = transparent / len(values)

    if transparent == 0:
        return "OPAQUE", ratio

    if transparent == len(values):
        return "TRANSPARENT", ratio

    return "MIXED", ratio

# ============================================================
# Subdivide boundary regions
# ============================================================

initial_faces = len(bm.faces)

for level in range(SUBDIVISION_LEVELS):
    mixed_faces = []

    for face in bm.faces:
        state, ratio = face_state(face)

        if state == "MIXED":
            mixed_faces.append(face)

    if not mixed_faces:
        print(f"Level {level + 1}: no mixed faces")
        break

    edges = set()

    for face in mixed_faces:
        edges.update(face.edges)

    print(
        f"Level {level + 1}: "
        f"subdividing {len(mixed_faces)} "
        f"boundary faces"
    )

    bmesh.ops.subdivide_edges(
        bm,
        edges=list(edges),
        cuts=1,
        use_grid_fill=True,
        smooth=0.0
    )

# ============================================================
# Remove transparent faces
# ============================================================

delete_faces = []

for face in bm.faces:
    state, ratio = face_state(face)

    if (
        state == "TRANSPARENT"
        or (
            state == "MIXED"
            and ratio >= DELETE_RATIO
        )
    ):
        delete_faces.append(face)

print(f"Original faces: {initial_faces}")
print(f"Faces before deletion: {len(bm.faces)}")
print(f"Faces to delete: {len(delete_faces)}")

if delete_faces:
    bmesh.ops.delete(
        bm,
        geom=delete_faces,
        context="FACES"
    )

# Remove loose edges and vertices after face deletion.
loose_edges = [
    e for e in bm.edges if not e.link_faces
]
if loose_edges:
    bmesh.ops.delete(
        bm, geom=loose_edges, context="EDGES"
    )

loose_verts = [
    v for v in bm.verts if not v.link_edges
]
if loose_verts:
    bmesh.ops.delete(
        bm, geom=loose_verts, context="VERTS"
    )

# ============================================================
# Write result
# ============================================================

bm.to_mesh(obj.data)
bm.free()

obj.data.update()

# Hide original, display result.
src.hide_set(True)

bpy.context.view_layer.objects.active = obj
obj.select_set(True)

print("\n=== Finished ===")
print(f"Output object: {obj.name}")
print(f"Remaining faces: {len(obj.data.polygons)}")
print("Original object preserved and hidden.")
