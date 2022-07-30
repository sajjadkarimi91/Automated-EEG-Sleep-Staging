function [wavelet_time_series,wavelet_time_series_n] = extract_wavelet_energys(EEG , srate , window , step )


window   = window * srate;                    % window length is 8 seconds
step     = step * srate;                    % step size is 2 seconds
windowNb = (length(EEG)-window)/step + 1;  % total number of windows(estimates)

gauss_window = gausswin(window , 0.75);
gauss_window = gauss_window(:)';

for n = 1:size(EEG,1)
    for i =   1  :  windowNb
        %         i
        curSegment = (i-1)*step+1 : (i-1)*step+window;

        temp_eeg = EEG(n , curSegment );
        temp_eeg = gauss_window.*temp_eeg;

        bands_freqs = [1,4,8,12,min(srate/2-0.1,30)];
        for k=1:4
            EEG_filtered = sjk_eeg_filter(temp_eeg,srate,bands_freqs(k),bands_freqs(k+1));
            energy_vector(k,1) = std(EEG_filtered);
        end
        wavelet_time_series((n-1)*4+1:n*4 , i ) = energy_vector;
        wavelet_time_series_n((n-1)*4+1:n*4 , i ) = energy_vector/sum(energy_vector);

    end

end


end




