
dir_path = "/gpfs/milgram/pi/chang/pg496/data_dir/social_gaze/social_gaze_eyetracking/rois";
mat_files = dir( fullfile(dir_path, "*position*.mat") );
out_path = "/gpfs/milgram/pi/chang/pg496/data_dir/social_gaze/social_gaze_eyetracking/roi_rect_tables";
ensurePathExists(out_path);
rois_of_interest = {"face", ...
    "eyes_nf", ...
    "mouth", ...
    "left_nonsocial_object", ...
    "right_nonsocial_object"};
for i = 1:numel(mat_files)
    fprintf("File: %d/%d\n", i, numel(mat_files));
    f_name = mat_files(i).name;
    roi_struct = load(fullfile(dir_path, f_name));
    roi_struct = roi_struct.var;
    m1_rois = roi_struct.m1.rects;
    roi_rects = struct();
    m1_roi_rects = convert_roi_map_to_struct(m1_rois, rois_of_interest);
    roi_rects.m1 = m1_roi_rects;
    out_f_path = fullfile(out_path, f_name);
    save(out_f_path, 'roi_rects');
    try
        m2_rois = roi_struct.m2.rects;
        m2_roi_rects = convert_roi_map_to_struct(m2_rois, rois_of_interest);
        roi_rects.m2 = m2_roi_rects;
        save(out_f_path, 'roi_rects');
    catch ME
        % If an error occurs, display the error message and continue to the next iteration
        fprintf('\nError processing file "%s": %s\n', f_name, ME.message);
        continue;
    end
end


function ensurePathExists(out_path)
    % Check if the directory already exists
    if exist(out_path, 'dir')
        disp(['Directory "', out_path, '" already exists.']);
    else
        % Attempt to create the directory
        try
            mkdir(out_path);
            disp(['Directory "', out_path, '" created successfully.']);
        catch
            % Display an error message if creation fails
            error(['Failed to create directory "', out_path, '".']);
        end
    end
end

function roiStruct = convert_roi_map_to_struct(roi_rect_map, rois_of_interest)
    % Initialize empty struct to store fields and values
    roiStruct = struct();
    % Iterate over each ROI of interest
    for i = 1:numel(rois_of_interest)
        roi = rois_of_interest{i};
        % Check if the ROI exists in the struct
        if ismember(roi, roi_rect_map.keys())
            % If the field exists, add it to roiStruct
            roiStruct.(roi) = roi_rect_map(roi);
        else
            warning('ROI "%s" not found in the struct.', roi);
            % Fill missing fields with NaN or any other placeholder
            roiStruct.(roi) = NaN;
        end
    end 
end


