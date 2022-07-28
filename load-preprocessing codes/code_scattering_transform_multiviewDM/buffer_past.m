function [frame_box] = buffer_past(signal,samplingrate,sec,boundary)

N = length(signal);
if (mod(N,samplingrate*sec)~=0)
    error('error')
end

frame_width = samplingrate*sec;
if (boundary == 1)
    frame_number = N/samplingrate/sec-4;
    frame_box = zeros(5*frame_width,frame_number);
    for k = 1:frame_number
    index = ((k-1)*frame_width+1):((k+4)*frame_width);
    frame_box(:,k) = signal(index);
    end
elseif (boundary == 0)
    frame_number = N/samplingrate/sec-2;
    frame_box = zeros(3*frame_width,frame_number);   
    for k = 1:frame_number
    index = ((k-1)*frame_width+1):((k+2)*frame_width);
    frame_box(:,k) = signal(index);
    end
end



end