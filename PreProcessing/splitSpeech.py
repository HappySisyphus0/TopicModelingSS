import os
import glob

# Input and output directories
input_dir = "hein-daily/Speeches"  # Update with correct path
output_dir = "groupedSpeeches"  # Update with desired output path
os.makedirs(output_dir, exist_ok=True)

# Define chunk size (adjust based on token length; this is a character limit for simplicity)
MAX_CHUNK_SIZE = 4000  # ~4096 tokens (~4000 characters)
def split_and_group_by_document():
    file_paths = glob.glob(os.path.join(input_dir, "*_097.txt"))  # Adjust to your files' format

    for file_path in file_paths:
        with open(file_path, "r", encoding="latin1") as f:
            text = f.read()

        # Split the text by "|", ensuring no speech is cut halfway
        speeches = text.split("|")

        # Create a directory for each input file
        file_name = os.path.basename(file_path).split(".")[0]  # Use the file name without extension
        file_output_dir = os.path.join(output_dir, file_name)
        os.makedirs(file_output_dir, exist_ok=True)

        current_chunk = ""
        current_chunk_size = 0
        chunk_number = 1

        # Iterate over each speech in the file
        for speech in speeches:
            speech = speech.strip()
            
            if speech:
                # Check if adding this speech would exceed the chunk size
                if current_chunk_size + len(speech) > MAX_CHUNK_SIZE:
                    # If chunk size is exceeded, save the current chunk
                    output_file = os.path.join(file_output_dir, f"chunk_{chunk_number}.txt")
                    with open(output_file, "w", encoding="latin1") as out_f:
                        out_f.write(current_chunk)
                    print(f"Created {output_file}")

                    # Start a new chunk with the current speech
                    current_chunk = speech
                    current_chunk_size = len(speech)
                    chunk_number += 1
                else:
                    # Add the speech to the current chunk
                    current_chunk += " | " + speech
                    current_chunk_size += len(speech)

        # Save the last chunk if it has content
        if current_chunk:
            output_file = os.path.join(file_output_dir, f"chunk_{chunk_number}.txt")
            with open(output_file, "w", encoding="latin1") as out_f:
                out_f.write(current_chunk)
            print(f"Created {output_file}")

    print("Speech splitting and grouping complete.")

# Run the splitting and grouping process
split_and_group_by_document()