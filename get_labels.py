def extract_labels(content_file, output_file):
    with open(content_file, 'r') as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        parts = line.strip().split()
        class_label = parts[-1]  # class_label is the last part
        labels.append(class_label)

    with open(output_file, 'w') as f:
        for label in labels:
            f.write(label + '\n')

if __name__ == '__main__':
    content_file = 'data/cora/cora.content'  
    output_file = 'data/cora/label.txt'      
    extract_labels(content_file, output_file)
    print(f"Labels extracted to {output_file}")
