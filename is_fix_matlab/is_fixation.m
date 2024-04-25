
function fix_vector = is_fixation(pos,time, t1, t2, minDur)

%  pos = aligned.m2.position;
%  time = aligned.m2.time;
data = [pos' time'];

if ( nargin < 5 )
  minDur = 0.01;
end

if ( nargin < 4 )
  t2 = 15;
end

if ( nargin < 3 )
  t1 = 30;
end

% t1 = 30;
% t2 = 15;
% minDur = 0.01; % 10 ms

dt = time(2)-time(1); 
% if start with pos, not NaN, add NaN
if isnan(data(1,1)) == 0
   newpoint = [NaN NaN time(1) - dt];
   data = [newpoint; data];
end
% if end with pos, not NaN, add NaN
if isnan(data(1,end)) == 0
   newpoint = [NaN NaN time(end) + dt];
   data = [data; newpoint];
end   
% data_length = length(data);
fix_vector = zeros(size(data,1),1);
diff_vector = diff(isnan(data(:,1)));
end_idc = find(diff_vector == 1);
start_idc = find(diff_vector == -1)+1;
% size_choice = size(end_idc,1)+1;

for i_segment = 1:size(start_idc,1)
    
    segment = [start_idc(i_segment) : end_idc(i_segment)];
 
    subdata = data(segment,:);
    [t_ind] = fixation_detection2(subdata,t1,t2,minDur,segment(1));
    for i = 1:numel(t_ind)
%         t_ind{i}(1)
        fix_vector(t_ind{i}(1):t_ind{i}(2)) = 1;
    end    
end    




