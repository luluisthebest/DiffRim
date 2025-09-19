clc
clearvars
close all

% constant parameters
c = physconst('LightSpeed');     % Speed of light in air (m/s)
fc = 77e9; % Center frequency (Hz)
lambda = freq2wavelen(fc,c); 
Rx = 4;
Tx = 2;

% configuration parameters
samples = 64;      
loop = 128;

fft_Rang = 64;
fft_Vel = 128;
max_value = 1e+04; % normalization the maximum of data WITH 1843

% Creat grid table
Range_max = 62.45;
Vel_max = 23.05;
rng_grid = (0:fft_Rang-1).'*(Range_max/fft_Rang);  %Fs*Tc=Samples per chirp (Assume it's fft_Rang)
vel_grid = (0:fft_Vel-1).'*(2*Vel_max/fft_Vel);   % unit: m/s, v = lamda/4*[-fs,fs], dopgrid = [-fs/2,fs/2]
vel_grid = vel_grid-Vel_max;

% Victim radar
Tc = 5.12e-6;
Fs = 12500000;
Bw = 153.6e6;

waveform = phased.FMCWWaveform('SweepTime',Tc, 'SweepBandwidth',Bw,'SampleRate',Fs,'SweepDirection','Up');
if strcmp(waveform.SweepDirection,'Down')
    sweepSlope = -sweepSlope;
end 

xt = complex(zeros(samples,loop));   % transmit signal
for m = 1:loop
    % Transmit FMCW waveform
    sig = waveform();
    xt(:,m) = sig;

end

% Based on AWR1843
tx_ppower = db2pow(12)*1e-3;                     % in watts
tx_gain = 1;                           % in dB

rx_gain = 24;                          % in dB
rx_nf = 15;                                    % in dB

receiver = phased.ReceiverPreamp('Gain',rx_gain,'NoiseFigure',rx_nf,...
    'SampleRate',Fs);
%% specify data name and load data as variable data_frames

sb0_mat = zeros(64,128,4334*7);
sb_mat = zeros(64,128,4334*7);
for i=0:4333
    filename = num2str(i,'%06d');
    seq_name = ['E:\50m_collection\2020-07-25-10-55-17\radar_raw_cube\',filename,'.mat'];
    img_name = ['E:\50m_collection\2020-07-25-10-55-17\camera_images\',filename, '.jpg'];
    image = imread(img_name);
    figure('Visible','on')
    imshow(image)

    load(seq_name);
    radar_data = permute(radar, [3 2 1]);
    [Rangedata] = fft_range(radar_data(:,1,:),fft_Rang,1);
    Dopplerdata = fft_doppler(Rangedata, fft_Vel, 0);
    RD_radar = squeeze(abs(Dopplerdata));

%     RD_log = log10(RD_radar)./log10(max(RD_radar,[],'all'));
    
% compare w and dB
    RD_log = 10 * log10(RD_radar);   % dB
    figure('visible','on')
    set(gcf,'Position',[10,10,530,420])
    [axh] = surf(vel_grid,rng_grid,RD_radar);
    % view(0,90)
    axis([-23.05 23.05 2 63]);
    % grid off
    shading interp
    xlabel('Doppler Velocity (m/s)')
    ylabel('Range(meters)')
    colorbar
    %caxis([])

    figure('visible','on')
    set(gcf,'Position',[10,10,530,420])
    [axh] = surf(vel_grid,rng_grid,RD_log);
    % view(0,90)
    axis([-23.05 23.05 2 63]);
    % grid off
    shading interp
    xlabel('Doppler Velocity (m/s)')
    ylabel('Range(meters)')
    colorbar
    %caxis([0,1])
  
    %%  Add interference

    int_fc = roundn(0.4*rand(1)+76.8,-1)*1e9;
    inf_tc = (26*rand(1)+4)*1e-6;                   %[4,30]
    inf_tc = ceil(inf_tc*Fs)/Fs;
    inf_bw = (660*rand(1)+140)*1e6;                  %[140,800]

    inf_sweep_slope = inf_bw/inf_tc;
    
    inf_waveform = phased.FMCWWaveform('SweepTime',inf_tc ,'SweepBandwidth',inf_bw,...
        'SampleRate', Fs,'SweepDirection','Up');
    if strcmp(inf_waveform.SweepDirection,'Down')
        sweepSlope = -sweepSlope;
    end 
    
    inf_rg = randi([2,63]);              %Relative radical range
    inf_vl = -(23.05*2*rand(1));   %Relative radical velocity
    
    inf_transmitter = phased.Transmitter('PeakPower',tx_ppower,'Gain',tx_gain);
    
    inf_radarmotion = phased.Platform('InitialPosition',[inf_rg;0;0.5],...
        'Velocity',[inf_vl;0;0]);
    
    inf_channel = phased.FreeSpace('PropagationSpeed',c,...
        'OperatingFrequency',int_fc,'SampleRate',Fs,'TwoWayPropagation',false);
    
    inf_Nsweep = ceil(loop*Tc/inf_waveform.SweepTime);
    x_inf = [];
    
    for m = 1:inf_Nsweep
        % Update interference radar
        [inf_radar_pos,inf_radar_vel] = inf_radarmotion(inf_waveform.SweepTime);
    
        % Transmit FMCW waveform
        inf_sig = inf_waveform();
        inf_txsig = inf_transmitter(inf_sig);
    
        % Propagate the signal (one-way)
        inf_txsig = inf_channel(inf_txsig,inf_radar_pos,[0;0;0.5],inf_radar_vel,[0;0;0]);
    
        % Receive radar return
        inf_txsig = receiver(inf_txsig);
    
        x_inf = [x_inf;inf_txsig];
    end
    
    xi = x_inf(1:samples*loop,:);
    xi = reshape(xi,[samples,loop]);
    xlpf = complex(zeros(samples,loop));
    
    for m=1:loop
        xdch = dechirp(xi(:,m),xt(:,m));
        %xlpf(:,m) = lowpass(xdch, Fs/2, Fs);
        xlpf(:,m) = xdch;
    end
    
    xlpf = max_value*reshape(xlpf, [samples, 1, loop]);
    
    [Rangedata_inf] = fft_range(xlpf,fft_Rang,1);
    Dopplerdata_inf = fft_doppler(Rangedata_inf, fft_Vel, 0);
    RD_inf = squeeze(abs(Dopplerdata_inf));
    
    RD_inf = 1000*RD_inf;
      
    sir = [-5:5:25];
    index = 1;
    norm = 0;       % 0 for only convert to dB; 1 for convert and normalize
    for j=index:length(sir)
        sir_index = 10^(-sir(j)/20);
        inf_sgn = RD_radar +sir_index*RD_inf;    
        
        if norm == 1
%         Normalization by log(x)/log(max)
            inf_sgn = log10(inf_sgn)./log10(max(inf_sgn,[],'all'));
        else
        % Normalization by log(x)
            inf_sgn = 10 * log10(inf_sgn);  % dB
        end
        
%         Plot range-Doppler image
        figure('visible','on')
        set(gcf,'Position',[10,10,530,420])
        [axh] = surf(vel_grid,rng_grid,inf_sgn);
        % view(0,90)
        axis([-23.05 23.05 2 63]);
        % grid off
        shading interp
        xlabel('Doppler Velocity (m/s)')
        ylabel('Range(meters)')
        colorbar
        %caxis([0,1])

        sb0_mat(:,:,i*7+j) = RD_log;
        %sb0_mat(:,:,i*7+j) = RD_radar;
        sb_mat(:,:,i*7+j) = inf_sgn;    
    
    end
    disp(i);
end

save('E:\50m_collection\2020-07-25-10-55-17\RD_map_raw.mat', 'sb0_mat', 'sb_mat');

  
