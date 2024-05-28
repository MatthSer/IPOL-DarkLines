def read_points_from_file(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            x1, y1, x2, y2 = map(int, line.strip().split(' '))
            points.append((x1, y1, x2, y2))
    return points


def generate_svg(points, output_filename):
    svg_header_1 = "<?xml version=\"1.0\" standalone=\"no\"?>\n"
    svg_header_2 = "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\""
    svg_header_3 = "\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n"
    svg_header_4 = "<svg width=\"500px\" height=\"500px\" version=\"1.1\" "
    svg_header_5 = "xmlns=\"http://www.w3.org/2000/svg\" "
    svg_header_6 = "xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n"
    svg_footer = '''</svg>'''

    svg_elements = []
    for x1, y1, x2, y2 in points:
        svg_elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="red" />')

    svg_content = '\n'.join([svg_header_1 + svg_header_2 + svg_header_3 + svg_header_4 + svg_header_5 + svg_header_6] + svg_elements + [svg_footer])

    with open(output_filename, 'w') as file:
        file.write(svg_content)


# Main function to read points and generate SVG
# def main():
#     input_filename = 'test_lines.txt'
#     output_filename = 'test_lines.svg'
#
#     points = read_points_from_file(input_filename)
#     generate_svg(points, output_filename)
#     print(f"SVG file '{output_filename}' created successfully.")
#
#
# if __name__ == "__main__":
#     main()
