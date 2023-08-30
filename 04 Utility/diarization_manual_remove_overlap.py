def remove_overlap(
    original_audio_path: str,
    metadata_file_path: str,
    output_dir: str,
    output_format: str = '.wav'
    ):

    '''
    If used, will remove any overlapping speakers. Is deployed on the WHOLE AUDIO level (not the chunk level). 
    Requires the previous metadata csv file which then gets overwritten with the new chunks.
    Also needs to overwrite the original metadatafile. 

    NOTE: This code became obsolete once the Pyanote built in overlap function was used. 


    '''
    
    # Read in metadata file and sort by ascending chunk number
    metadata = pd.read_csv(metadata_file_path)
    metadata = metadata.sort_values('chunk_num')
    
    overlapping_sections = []
    non_overlapping_sections = []
    
    # Select clip of interest
    for id, clip in metadata.iterrows():
        
        clip_start = clip['start']
        clip_end = clip['end']
        
        mask = metadata.index != id
        other_clips = metadata[mask]
    
        clip_chunk = clip['chunk_num']

        # Check for overlap with all other clips 
        for id_x, clip_x in other_clips.iterrows():
            
            other_clip_chunk = clip_x['chunk_num']
            other_clip_speaker = clip_x['speaker']
    
            overlap = False
            overlap_start = clip_start
            overlap_end = clip_end
            
            # Check starting & ending condition overlap
            if clip_x['start'] > clip_start and clip_x['start'] < clip_end:
                overlap = True
                overlap_start = clip_x['start']
        
            if clip_x['end'] > clip_start and clip_x['end'] < clip_end:
                overlap = True
                overlap_end = clip_x['end']
    
            if overlap:
                overlapping_sections.append({'start':overlap_start, 'end':overlap_end})
    
    audio = AudioSegment.from_file(original_audio_path)
    
    # Convert timestamps to milliseconds
    new_audio = audio

    # Remove duplicated from overlapping sections AND SORT because they get caught twice in the double clip loop
    seen_items = set()
    overlapping_sections = [seen_items.add(item['start']) or item for item in overlapping_sections if item['start'] not in seen_items]
    overlapping_sections = sorted(overlapping_sections, key=lambda x: x['start'])
    
    # Original code for creating the full audio again without overlap - obsolete 
    for ts in overlapping_sections:
        ts['start'] = int(ts['start'])
        ts['end'] = int(ts['end'])
        new_audio = new_audio[:ts['start']] + new_audio[ts['end']:]

    #return new_audio # For original code if needed 

    # Generate the non_overlapping_sections from the overlapping ones 
    non_overlapping_sections = []
    
    for i in range(len(overlapping_sections)):
        if i == 0:
            non_overlap_start = 0
            non_overlap_end = overlapping_sections[i]['start']
            non_overlapping_sections.append({'start':non_overlap_start, 'end':non_overlap_end})
        else:
            try:
                non_overlap_start = overlapping_sections[i-1]['end']
                non_overlap_end = overlapping_sections[i]['start']
                non_overlapping_sections.append({'start':non_overlap_start, 'end':non_overlap_end})
            
            except:
                break

    # Allocate speaker to each new section 
    for i in range(min(len(overlapping_sections), len(non_overlapping_sections))):
        print(i)
        print(f'Non: {non_overlapping_sections[i]}')
        print(f'Over: {overlapping_sections[i]}')

    new_metadata = pd.DataFrame()
    i = 0

    file_name = original_audio_path.split('/')[-1]
    
    for chunk in overlapping_sections: 
        file = file_name
        output_name = os.path.basename(
                f'overlap_{file.replace(splitext(file)[1], "")}_chunk_{i}_{output_format}'
            )
        output_path = os.path.join(output_dir, output_name)
        speaker = ''
        start = chunk['start']
        end = chunk['end']
        new_metadata = pd.concat([
                    pd.DataFrame(
                        [[file, output_name, str(i), output_path, speaker, start, end, 1]],
                        columns=['source_name', 'file_name', 'chunk_num',
                                 'path', 'speaker', 'start', 'end', 'overlap']),
                    new_metadata
                ])
        i += 1

    for chunk in non_overlapping_sections: 
        file = file_name
        output_name = os.path.basename(
                f'non_overlap_{file.replace(splitext(file)[1], "")}_chunk_{i}_{output_format}'
            )
        output_path = os.path.join(output_dir, output_name)
        speaker = ''
        start = chunk['start']
        end = chunk['end']
        new_metadata = pd.concat([
                    pd.DataFrame(
                        [[file, output_name, str(i), output_path, speaker, start, end, 0]],
                        columns=['source_name', 'file_name', 'chunk_num',
                                 'path', 'speaker', 'start', 'end', 'overlap']),
                    new_metadata
                ])
        i += 1
    
    
    
    return new_audio, new_metadata