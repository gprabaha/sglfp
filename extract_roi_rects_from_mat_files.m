
dir_path = "/gpfs/milgram/pi/chang/pg496/data_dir/social_gaze/social_gaze_eyetracking/rois";
mat_files = dir( fullfile(dir_path, "*.mat") );
out_path = "/gpfs/milgram/pi/chang/pg496/data_dir/social_gaze/social_gaze_eyetracking/roi_rect_tables";
ensurePathExists(out_path);
rois_of_interest = {"face", ...
    "eyes_nf", ...
    "mouth", ...
    "left_nonsocial_object", ...
    "right_nonsocial_object"};
for i = 1:numel(mat_files)
    try
        f_name = mat_files(i).name;
        roi_struct = load(fullfile(dir_path, f_name));
        roi_struct = roi_struct.var;
        m1_rois = roi_struct.m1.rects;
        roi_rects = struct();
        m1_roi_rects = extractROIFields(m1_rois, rois_of_interest);
        roi_rects.m1 = m1_roi_rects;
        m2_rois = roi_struct.m2.rects;
        m2_roi_rects = extractROIFields(m2_rois, rois_of_interest);
        roi_rects.m2 = m2_roi_rects;
        out_f_path = fullfile(out_path, f_name);
        save(out_f_path, 'roi_rects');
        % Update progress bar
        progress = i / numel(mat_files) * 100;
        fprintf(repmat('\b', 1, 5));  % Erase previous percentage
        fprintf('%3.0f%%', progress);
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


function roiTable = extractROIFields(roi_rect_map, rois_of_interest)
    % Initialize empty cell arrays to store field names and values
    fieldNames = {};
    fieldValues = {};
    % Iterate over each ROI of interest
    for i = 1:numel(rois_of_interest)
        roi = rois_of_interest{i};
        % Check if the ROI exists in the struct
        if ismember(roi, roi_rect_map.keys() )
            % If the field exists, add its name to fieldNames
            fieldNames = [fieldNames; roi];
            % Get the value of the field and add it to fieldValues
            fieldValue = roi_rect_map(roi);
            fieldValues = [fieldValues; {fieldValue}];
        else
            warning('ROI "%s" not found in the struct.', roi);
        end
    end 
    % Create a table from fieldNames and fieldValues
    roiTable = table(fieldNames, fieldValues);
end
