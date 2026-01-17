"""
Sanity checks for NCD distance.
Tests pairwise distances for:
1. Same galaxy vs itself (should be ~0)
2. Similar galaxies (should be low)
3. Different galaxies (should be high)
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.compression import NCDCalculator, CanonicalImageProcessor
from src.preprocess import GalaxyDataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def create_test_image(size=(256, 256), shape='circle'):
    """Create synthetic test image."""
    from PIL import ImageDraw

    img = Image.new('L', size, color=0)
    draw = ImageDraw.Draw(img)

    if shape == 'circle':
        draw.ellipse([50, 50, 200, 200], fill=255)
    elif shape == 'square':
        draw.rectangle([50, 50, 200, 200], fill=255)
    elif shape == 'spiral':
        # Simple spiral approximation
        center_x, center_y = size[0] // 2, size[1] // 2
        for angle in np.linspace(0, 4 * np.pi, 100):
            r = 20 + angle * 5
            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=255)

    return img


def test_synthetic_images():
    """Test with synthetic images to verify NCD behavior."""
    print("=" * 60)
    print("Testing with synthetic images")
    print("=" * 60)

    # Create test images
    circle_img = create_test_image(shape='circle')
    square_img = create_test_image(shape='square')
    spiral_img = create_test_image(shape='spiral')

    # Save temporarily
    temp_dir = "temp_test"
    os.makedirs(temp_dir, exist_ok=True)

    circle_path = os.path.join(temp_dir, "circle.png")
    square_path = os.path.join(temp_dir, "square.png")
    spiral_path = os.path.join(temp_dir, "spiral.png")

    circle_img.save(circle_path)
    square_img.save(square_path)
    spiral_img.save(spiral_path)

    # Initialize calculator
    calculator = NCDCalculator()

    # Test 1: Same image
    dist_self, comp_size = calculator.compute_self_distance(circle_path)
    print(f"\n1. Circle vs itself:")
    print(f"   NCD = {dist_self:.6f}")
    print(f"   Compressed size = {comp_size} bytes")
    print(f"   ✓ Expected: Very close to 0 (got {dist_self:.6f})")

    # Test 2: Circle vs Square (similar compact shapes)
    dist_circle_square = calculator.ncd(circle_path, square_path)
    print(f"\n2. Circle vs Square:")
    print(f"   NCD = {dist_circle_square:.4f}")

    # Test 3: Circle vs Spiral (different shapes)
    dist_circle_spiral = calculator.ncd(circle_path, spiral_path)
    print(f"3. Circle vs Spiral:")
    print(f"   NCD = {dist_circle_spiral:.4f}")

    # Test 4: Square vs Spiral
    dist_square_spiral = calculator.ncd(square_path, spiral_path)
    print(f"4. Square vs Spiral:")
    print(f"   NCD = {dist_square_spiral:.4f}")

    # Verify ordering
    print(f"\n✓ Verification:")
    print(f"  Self distance ({dist_self:.4f}) < Similar shapes ({dist_circle_square:.4f})")
    print(f"  Similar shapes ({dist_circle_square:.4f}) < Different shapes ({dist_circle_spiral:.4f})")

    if dist_self < dist_circle_square < dist_circle_spiral:
        print("  ✓ All inequalities hold as expected!")
    else:
        print("  ⚠ Some inequalities don't hold - check implementation")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

    return {
        'self_distance': dist_self,
        'circle_square': dist_circle_square,
        'circle_spiral': dist_circle_spiral,
        'square_spiral': dist_square_spiral
    }


def test_real_galaxies(data_dir="data/raw"):
    """Test with real galaxy images from dataset."""
    print("\n" + "=" * 60)
    print("Testing with real galaxy images")
    print("=" * 60)

    dataset = GalaxyDataset(data_dir)

    try:
        dataset.load_images("*.jpg")
    except:
        # Try other extensions
        for ext in ["*.png", "*.jpeg", "*.tiff"]:
            dataset.load_images(ext)
            if dataset.image_paths:
                break

    if not dataset.image_paths:
        print("No images found in data/raw/")
        print("Please add some galaxy images to test.")
        return None

    print(f"Found {len(dataset.image_paths)} images")

    # Initialize calculator
    calculator = NCDCalculator()

    # Test cases
    test_cases = []

    # Case 1: Same image
    if len(dataset.image_paths) >= 1:
        img1 = dataset.image_paths[0]
        test_cases.append(("Same image", img1, img1))

    # Case 2: Potentially similar images (first two)
    if len(dataset.image_paths) >= 2:
        img1, img2 = dataset.image_paths[0], dataset.image_paths[1]
        test_cases.append(("First two images", img1, img2))

    # Case 3: Potentially different images (first and last)
    if len(dataset.image_paths) >= 3:
        img1, img2 = dataset.image_paths[0], dataset.image_paths[-1]
        test_cases.append(("First and last image", img1, img2))

    # Run tests
    results = {}

    for name, path1, path2 in test_cases:
        if path1 == path2:
            dist, comp_size = calculator.compute_self_distance(path1)
            print(f"\n{name}:")
            print(f"  Path: {os.path.basename(path1)}")
            print(f"  NCD = {dist:.6f}")
            print(f"  Compressed size = {comp_size} bytes")
            results[name] = dist
        else:
            dist = calculator.ncd(path1, path2)
            print(f"\n{name}:")
            print(f"  Image 1: {os.path.basename(path1)}")
            print(f"  Image 2: {os.path.basename(path2)}")
            print(f"  NCD = {dist:.4f}")
            results[name] = dist

    return results


def visualize_test_results(synthetic_results, real_results=None):
    """Create visualization of test results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Synthetic results
    if synthetic_results:
        synthetic_labels = ['Self\n(circle)', 'Circle vs\nSquare', 'Circle vs\nSpiral', 'Square vs\nSpiral']
        synthetic_values = [
            synthetic_results['self_distance'],
            synthetic_results['circle_square'],
            synthetic_results['circle_spiral'],
            synthetic_results['square_spiral']
        ]

        bars = axes[0].bar(synthetic_labels, synthetic_values)
        axes[0].set_title('NCD for Synthetic Shapes')
        axes[0].set_ylabel('NCD Distance')
        axes[0].grid(True, alpha=0.3)

        # Color bars by value
        for bar, value in zip(bars, synthetic_values):
            if value < 0.1:
                bar.set_color('green')
            elif value < 0.3:
                bar.set_color('orange')
            else:
                bar.set_color('red')

    # Real results
    if real_results:
        real_labels = list(real_results.keys())
        real_values = list(real_results.values())

        # Truncate long labels
        real_labels_short = [label[:15] + '...' if len(label) > 15 else label
                             for label in real_labels]

        bars = axes[1].bar(real_labels_short, real_values)
        axes[1].set_title('NCD for Real Galaxies')
        axes[1].set_ylabel('NCD Distance')
        axes[1].grid(True, alpha=0.3)

        # Color bars
        for bar, value in zip(bars, real_values):
            if value < 0.1:
                bar.set_color('green')
            elif value < 0.3:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        # Rotate x-axis labels for readability
        plt.sca(axes[1])
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Save figure
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/sanity_checks.png", dpi=150, bbox_inches='tight')
    plt.savefig("results/figures/sanity_checks.pdf", bbox_inches='tight')

    plt.show()


def main():
    """Run all sanity checks."""
    print("NCD Distance Sanity Checks")
    print("=" * 60)

    # Test 1: Synthetic images
    synthetic_results = test_synthetic_images()

    # Test 2: Real galaxies (if available)
    real_results = test_real_galaxies()

    # Visualize results
    visualize_test_results(synthetic_results, real_results)

    # Save numerical results
    import json
    os.makedirs("results/distances", exist_ok=True)

    all_results = {
        'synthetic': synthetic_results,
        'real': real_results if real_results else "No real images found"
    }

    with open("results/distances/sanity_check_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("Results saved to:")
    print("  - results/figures/sanity_checks.png")
    print("  - results/distances/sanity_check_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()