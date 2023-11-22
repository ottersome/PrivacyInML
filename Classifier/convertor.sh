#!/bin/sh

# This script parses a tree and creates a new tree with the same structure
# but different names and files with new format

new_name(){ echo "$1" | sed 's|\([^/]*\)\/|\1_new\/|1';}
get_subdirs() { find $1  -mindepth 1 -type d;}
get_file() { ls "$1/.*.$2" 2> /dev/null;}
get_formats() { find $1 -type f -name "*.$2"; }

root_dir=$1
file_types=$2
conversion_command=$3
# Create Dirs first
for dir in $(get_subdirs $1)
do
  new_name=$(new_name $dir)
  echo "Creating dir $new_name"
  mkdir -p $new_name
done

# Then do conversion and send it to a differetn dir
for filo in $(get_formats $root_dir $file_types)
do
  new_name=$(new_name $filo | sed "s|\.$file_types|.png|g")
  echo "Copying $new_name ..."
  convert $filo $new_name
done

