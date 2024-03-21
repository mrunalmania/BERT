# In this file we download the cornell database from the official cornell website


# Download the file
Invoke-WebRequest -Uri 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip' -OutFile 'cornell_movie_dialogs_corpus.zip'

# Extract the zip file
Expand-Archive -Path 'cornell_movie_dialogs_corpus.zip' -DestinationPath 'datasets'

# Remove the zip file
Remove-Item 'cornell_movie_dialogs_corpus.zip'

# Create a directory
New-Item -ItemType Directory -Path 'datasets' -Force

# Move files to the datasets directory
Move-Item -Path 'cornell movie-dialogs corpus/movie_conversations.txt' -Destination 'datasets' -Force
Move-Item -Path 'cornell movie-dialogs corpus/movie_lines.txt' -Destination 'datasets' -Force