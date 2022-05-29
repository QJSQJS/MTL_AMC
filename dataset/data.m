%%%%%%%%%%%%%%%%%%%%%%%%%%%鐟炲埄淇￠亾%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;
close all;
% Create binary data for 100, 4-bit symbols
% M_SNR_dB                  = [30,32];              % Signal-to-Noise Ratio in dB
M_SNR_dB                  = [-20:2:18];              % Signal-to-Noise Ratio in dB
NrRepetitions             = 1000;                    % 姣忎竴涓皟鍒舵柟寮忔瘡涓�涓猄NR涓嬬殑鏍锋湰鏁�  

single_PAM4 = []; single_QAM16 = []; single_QAM64 = []; single_CPFSK = []; single_BPSK = []; single_QPSK = [];
single_my8PSK = [];single_FM = [];single_AM_DSB = [];single_AM_SSB = [];single_GFSK = [];

single_PAM4_h = []; single_QAM16_h = []; single_QAM64_h = []; single_CPFSK_h = []; single_BPSK_h = []; single_QPSK_h = [];
single_my8PSK_h = [];single_FM_h = [];single_AM_DSB_h = [];single_AM_SSB_h = [];single_GFSK_h = [];

single_PAM4_noise = []; single_QAM16_noise = []; single_QAM64_noise = []; single_CPFSK_noise = []; 
single_BPSK_noise = []; single_QPSK_noise = [];single_my8PSK_noise = [];
single_FM_noise = [];single_AM_DSB_noise = [];single_AM_SSB_noise = [];single_GFSK_noise = [];

disp('dataset began');star=datestr(now,31);disp(star);

for i_rep = 1:NrRepetitions
    data = randi([0 1],256,1);
    
    data_PAM4 = randi([0 1],512,1);
    PAM4 = comm.PAMModulator(4,'BitInput',true);
    PAM4_x = step(PAM4,data_PAM4);
    
    data_QAM16 = randi([0 1],1024,1);
    QAM16 = comm.RectangularQAMModulator('ModulationOrder',16,'BitInput',true);   
    QAM16_x = step(QAM16, data_QAM16);
    
    data_CPFSK = randi([0 1],96,1);
    CPFSK = comm.CPFSKModulator(8, 'BitInput', true, 'SymbolMapping', 'Gray');
    CPFSK_x = step(CPFSK,data_CPFSK);

    data_QAM64 = randi([0 1],1536,1);
    QAM64 = comm.RectangularQAMModulator('ModulationOrder',64,'BitInput',true);
    QAM64_x = step(QAM64, data_QAM64);
    
    BPSK = comm.BPSKModulator;
    BPSK_x = step(BPSK,data);
    
    data2 = randi([0 3],256,1);
    QPSK = comm.QPSKModulator;
    QPSK_x = step(QPSK,data2);
    
    data3 = randi([0 7],256,1);
    my8PSK = comm.PSKModulator;
    my8PSK_x = step(my8PSK,data3);
    
    fs = 10000; 
    fc = 4000;  
    t = (0:1/fs:0.0255)';
    x = sin(2*pi*30*t)+2*sin(2*pi*60*t);
    fDev = 50;
    FM_x = fmmod(x,fc,fs,fDev);
    
    fs = 10000;
    t = (0:1/fs:0.0255)';
    fc = 4000;
    x = sin(2*pi*30*t)+2*sin(2*pi*60*t);
    AM_DSB_x = ammod(x,fc,fs); AM_SSB_x = ssbmod(x,fc,fs);
    
%     data = randi([0 1],16,1);
%     GFSK_x = lrwpan.PHYGeneratorGFSK(data,4);

% scatterplot(QAM16_x);scatterplot(QAM64_x);scatterplot(CPFSK_x);
% scatterplot(BPSK_x);scatterplot(QPSK_x);scatterplot(my8PSK_x);
% scatterplot(FM_x);scatterplot(AM_DSB_x);scatterplot(AM_SSB_x);scatterplot(PAM4_x);
%   
        
    for i_SNR = 1:length(M_SNR_dB)
        
%         % 浣跨敤 comm.AWGNChannel 瀹炵幇
% awgnChannel = comm.AWGNChannel(...
%   'NoiseMethod', 'Signal to noise ratio (SNR)', ...
%   'SignalPower', 1, ...
%   'SNR', i_SNR);
% 
% %% 1.1.2 鑾辨柉澶氬緞琛拌惤
fs = 200e3;             % Sample rate
    multipathChannel = comm.RicianChannel(...
    'SampleRate',200e3,...
    'PathDelays',[0.0 0.9 1.7],...
    'AveragePathGains',[0.1 0.5 0.2],...
    'KFactor',4,...
    'DirectPathDopplerShift',50,...
    'DirectPathInitialPhase',5,...
    'MaximumDopplerShift',4,...
    'DopplerSpectrum',doppler('Bell', 8),...
    'PathGainsOutputPort',true);

% % 浣跨敤 comm.RicianChannel System object 瀹炵幇閫氳繃鑾辨柉澶氬緞琛拌惤淇￠亾銆傚亣璁惧欢杩熷垎甯冧负 [0 1.8 3.4] 涓牱鏈紝瀵瑰簲鐨勫钩鍧囪矾寰勫鐩婁负 [0 -2 -10] dB銆侹 鍥犲瓙涓� 4锛屾渶澶у鏅嫆棰戠Щ涓� 4 Hz锛岀瓑鏁堜簬 900 MHz 鐨勬琛岄�熷害銆備娇鐢ㄤ互涓嬭缃疄鐜颁俊閬撱��
% multipathChannel = comm.RicianChannel(...
%   'SampleRate', fs, ...
%   'PathDelays', [0 1.8 3.4]/fs, ...
%   'AveragePathGains', [0 -2 -10], ...
%   'KFactor', 4, ...
%   'MaximumDopplerShift', 4);
% 
%% 1.1.3 鏃堕挓鍋忕Щ
% .鏃堕挓鍋忕Щ鏄彂閫佸櫒鍜屾帴鏀跺櫒鐨勫唴閮ㄦ椂閽熸簮涓嶅噯纭�犳垚鐨勩�傛椂閽熷亸绉诲鑷翠腑蹇冮鐜囷紙鐢ㄤ簬灏嗕俊鍙蜂笅鍙橀鑷冲熀甯︼級鍜屾暟妯¤浆鎹㈠櫒閲囨牱鐜囦笉鍚屼簬鐞嗘兂鍊笺�備俊閬撲豢鐪熷櫒浣跨敤鏃堕挓鍋忕Щ鍥犲瓙 C锛岃〃绀轰负 C=1+螖clock106锛屽叾涓� 螖clock 鏄椂閽熷亸绉汇�傚浜庢瘡涓抚锛岄�氶亾鍩轰簬 [?max螖clock max螖clock] 鑼冨洿鍐呬竴缁勫潎鍖�鍒嗗竷鐨勫�肩敓鎴愪竴涓殢鏈� 螖clock 鍊硷紝鍏朵腑 max螖clock 鏄渶澶ф椂閽熷亸绉汇�傛椂閽熷亸绉讳互鐧句竾鍒嗙巼 (ppm) 涓哄崟浣嶆祴閲忋�傚浜庢湰绀轰緥锛屽亣璁炬渶澶ф椂閽熷亸绉讳负 5 ppm銆�
maxDeltaOff = 0.5;
deltaOff = (rand()*2*maxDeltaOff) - maxDeltaOff;
C = 1 + (deltaOff/1e6);
% 棰戠巼鍋忕Щ
% 鍩轰簬鏃堕挓鍋忕Щ鍥犲瓙 C 鍜屼腑蹇冮鐜囷紝瀵规瘡甯ц繘琛岄鐜囧亸绉汇�備娇鐢� comm.PhaseFrequencyOffset 瀹炵幇淇￠亾銆�
offset = -(C-1)*fc(1);
frequencyShifter = comm.PhaseFrequencyOffset(...
  'SampleRate', fs, ...
  'FrequencyOffset', offset);
% 閲囨牱鐜囧亸绉�
% 鍩轰簬鏃堕挓鍋忕Щ鍥犲瓙 C锛屽姣忓抚杩涜閲囨牱鐜囧亸绉汇�備娇鐢� interp1 鍑芥暟瀹炵幇閫氶亾锛屼互 C脳fs 鐨勬柊閫熺巼瀵瑰抚杩涜閲嶆柊閲囨牱銆�
% 
%% 1.1.4 鍚堟垚鍚庣殑淇￠亾
% 浣跨敤 helperModClassTestChannel 瀵硅薄瀵瑰抚搴旂敤鎵�鏈変笁绉嶄俊閬撹“钀�
channel = helperModClassTestChannel(...
  'SampleRate', fs, ...
  'SNR', 10000000, ...
  'PathDelays', [0 0.9 1.7] / fs, ...
  'AveragePathGains', [0.1 0.5 0.2], ...
  'KFactor', 4, ...
  'MaximumDopplerShift', 4, ...
  'MaximumClockOffset', 0.5, ...
  'CenterFrequency', 902e6);
% % 鎮ㄥ彲浠ヤ娇鐢� info 瀵硅薄鍑芥暟鏌ョ湅鏈夊叧閫氶亾鐨勫熀鏈俊鎭��

% chInfo = info(channel);

        channel.CenterFrequency = 902e6;
        PAM4_xh = channel(PAM4_x);QAM16_xh = channel(QAM16_x);QAM64_xh = channel(QAM64_x);
        CPFSK_xh = channel(CPFSK_x);BPSK_xh = channel(BPSK_x);QPSK_xh = channel(QPSK_x);my8PSK_xh = channel(my8PSK_x);
        
        channel.CenterFrequency = 100e6;
        FM_xh = channel(FM_x);AM_DSB_xh = channel(AM_DSB_x);AM_SSB_xh = channel(AM_SSB_x);
        
        %鍘婚櫎NAN鍊�
        PAM4_xh(isnan(PAM4_xh)) = 0;QAM16_xh(isnan(QAM16_xh)) = 0;QAM64_xh(isnan(QAM64_xh)) = 0;
        CPFSK_xh(isnan(CPFSK_xh)) = 0;BPSK_xh(isnan(BPSK_xh)) = 0;QPSK_xh(isnan(QPSK_xh)) = 0;
        my8PSK_xh(isnan(my8PSK_xh)) = 0;
        FM_xh(isnan(FM_xh)) = 0;AM_DSB_xh(isnan(AM_DSB_xh)) = 0;AM_SSB_xh(isnan(AM_SSB_xh)) = 0;
%         GFSK_xhn = channel(GFSK_x);

scatterplot(QAM16_xh);scatterplot(QAM64_xh);scatterplot(CPFSK_xh);scatterplot(BPSK_xh);
scatterplot(QPSK_xh);scatterplot(my8PSK_xh);scatterplot(FM_xh);scatterplot(AM_DSB_xh);
scatterplot(AM_SSB_xh);% scatterplot(GFSK_xh);    
scatterplot(PAM4_xh);


        PAM4_xhn = awgn(PAM4_xh,i_SNR);QAM16_xhn = awgn(QAM16_xh, i_SNR);QAM64_xhn = awgn(QAM64_xh, i_SNR);
        CPFSK_xhn = awgn(CPFSK_xh, i_SNR);BPSK_xhn = awgn(BPSK_xh, i_SNR);QPSK_xhn = awgn(QPSK_xh, i_SNR);
        my8PSK_xhn = awgn(my8PSK_xh, i_SNR);FM_xhn = awgn(FM_xh, i_SNR);AM_DSB_xhn = awgn(AM_DSB_xh, i_SNR);AM_SSB_xhn = awgn(AM_SSB_xh, i_SNR);
%         GFSK_xhn = awgn(GFSK_xh, i_SNR);

scatterplot(QAM16_xhn);scatterplot(QAM64_xhn);scatterplot(CPFSK_xhn);scatterplot(BPSK_xhn);
scatterplot(QPSK_xhn);scatterplot(my8PSK_xhn);scatterplot(FM_xhn);scatterplot(AM_DSB_xhn);
scatterplot(AM_SSB_xhn);% scatterplot(GFSK_xhn);    
scatterplot(PAM4_xhn);
        
        single_PAM4(i_rep,i_SNR,:) =PAM4_x; single_QAM16(i_rep,i_SNR,:) = QAM16_x;
        single_QAM64(i_rep,i_SNR,:) = QAM64_x;
        single_CPFSK(i_rep,i_SNR,:) =CPFSK_x; single_BPSK(i_rep,i_SNR,:) = BPSK_x;
        single_QPSK(i_rep,i_SNR,:) =QPSK_x; single_my8PSK(i_rep,i_SNR,:) = my8PSK_x;
        single_FM(i_rep,i_SNR,:) =FM_x; single_AM_DSB(i_rep,i_SNR,:) = AM_DSB_x;
        single_AM_SSB(i_rep,i_SNR,:) =AM_SSB_x; 
%         single_GFSK(i_rep,i_SNR,:) = GFSK_x;
        
        single_PAM4_h(i_rep,i_SNR,:) =PAM4_xh; single_QAM16_h(i_rep,i_SNR,:) = QAM16_xh;
        single_QAM64_h(i_rep,i_SNR,:) = QAM64_xh;
        single_CPFSK_h(i_rep,i_SNR,:) =CPFSK_xh; single_BPSK_h(i_rep,i_SNR,:) = BPSK_xh;
        single_QPSK_h(i_rep,i_SNR,:) =QPSK_xh; single_my8PSK_h(i_rep,i_SNR,:) = my8PSK_xh;
        single_FM_h(i_rep,i_SNR,:) =FM_xh; single_AM_DSB_h(i_rep,i_SNR,:) = AM_DSB_xh;
        single_AM_SSB_h(i_rep,i_SNR,:) =AM_SSB_xh; 
%         single_GFSK(i_rep,i_SNR,:) = GFSK_x;
        
        single_PAM4_noise(i_rep,i_SNR,:) =PAM4_xhn; single_QAM16_noise(i_rep,i_SNR,:) = QAM16_xhn;
        single_QAM64_noise(i_rep,i_SNR,:) = QAM64_xhn;
        single_CPFSK_noise(i_rep,i_SNR,:) =CPFSK_xhn; single_BPSK_noise(i_rep,i_SNR,:) = BPSK_xhn;
        single_QPSK_noise(i_rep,i_SNR,:) =QPSK_xhn; single_my8PSK_noise(i_rep,i_SNR,:) = my8PSK_xhn;
        single_FM_noise(i_rep,i_SNR,:) =FM_xhn; single_AM_DSB_noise(i_rep,i_SNR,:) = AM_DSB_xhn;
        single_AM_SSB_noise(i_rep,i_SNR,:) =AM_SSB_xhn; 
%         single_GFSK_noise(i_rep,i_SNR,:) = GFSK_xhn;
        

    end
end  

save single_PAM4.mat single_PAM4
save single_QAM16.mat single_QAM16
save single_QAM64.mat single_QAM64
save single_CPFSK.mat single_CPFSK
save single_BPSK.mat single_BPSK
save single_QPSK.mat single_QPSK
save single_my8PSK.mat single_my8PSK
save single_FM.mat single_FM
save single_AM_DSB.mat single_AM_DSB
save single_AM_SSB.mat single_AM_SSB
% save single_GFSK.mat single_GFSK

save single_PAM4_h.mat single_PAM4_h
save single_QAM16_h.mat single_QAM16_h
save single_QAM64_h.mat single_QAM64_h
save single_CPFSK_h.mat single_CPFSK_h
save single_BPSK_h.mat single_BPSK_h
save single_QPSK_h.mat single_QPSK_h
save single_my8PSK_h.mat single_my8PSK_h
save single_FM_h.mat single_FM_h
save single_AM_DSB_h.mat single_AM_DSB_h
save single_AM_SSB_h.mat single_AM_SSB_h
% save single_GFSK_h.mat single_GFSK_h

save single_PAM4_hnoise.mat single_PAM4_noise
save single_QAM16_hnoise.mat single_QAM16_noise
save single_QAM64_hnoise.mat single_QAM64_noise
save single_CPFSK_hnoise.mat single_CPFSK_noise
save single_BPSK_hnoise.mat single_BPSK_noise
save single_QPSK_hnoise.mat single_QPSK_noise
save single_my8PSK_hnoise.mat single_my8PSK_noise
save single_FM_hnoise.mat single_FM_noise
save single_AM_DSB_hnoise.mat single_AM_DSB_noise
save single_AM_SSB_hnoise.mat single_AM_SSB_noise
% save single_GFSK_hnoise.mat single_GFSK_noise


disp('dataset ok');down=datestr(now,31);disp(down);




