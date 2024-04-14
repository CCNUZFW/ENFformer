close all;clear all;clc
% [~,txto,~] = xlsread('database\WHU-h1-or-1000.xlsx');
% [~,txte,~] = xlsread('database\WHU-h1-ed-1000.xlsx');
[~,txto,~] = xlsread('database\WHU_H1_or_2584.xlsx');
[~,txte,~] = xlsread('database\WHU_H1_ed_2584.xls');
DFT0_DFT1_Hilbert_time=[];

ppmax = 2055;
ffmax = 37636;

for i = 1:length(txto)
    filename1=[txto{i}];
    filename2=[txte{i}];
    
    [F1o,F2o,po0,po]=gf_50Hz(filename1,10,2000);
    [F3o,fo]=Ex_hilbertIF_F3(filename1,50);

    [F1e,F2e,pe0,pe]=gf_50Hz(filename2,10,2000);
    [F3e,fe]=Ex_hilbertIF_F3(filename2,50);
    
    %时序相位表征
    [fram_num0,fram_len0,po1_res0] = anew_fram_encode(po0,25,'next','fram_len',ppmax);%未篡改DFT0相位时序特征
    [~,~,pe1_res0] = anew_fram_encode(pe0,25,'next','fram_len',ppmax);%篡改DFT0相位时序特征
    [fram_num,fram_len,po1_res] = anew_fram_encode(po,25,'next','fram_len',ppmax);%未篡改DFT1相位时序特征
    [~,~,pe1_res] = anew_fram_encode(pe,25,'next','fram_len',ppmax);%篡改DFT1相位时序特征
    [fram_num1,fram_len1,fo1_res] = anew_fram_encode(fo,256,'next','fram_len',ffmax);%未篡改频率时序特征
    [~,~,fe1_res] = anew_fram_encode(fe,256,'next','fram_len',ffmax);%篡改频率时序特征

    i;
    fram_num0;
    fram_len0;
    fram_num;
    fram_len;
    fram_num1;
    fram_len1;

    DFT0_DFT1_Hilbert_time=[DFT0_DFT1_Hilbert_time;po1_res0,po1_res,fo1_res,1;pe1_res0,pe1_res,fe1_res,0];

end

% dlmwrite('generate_data\DFT0_DFT1_25_83_Hilbert_256_148_2000.txt', DFT0_DFT1_Hilbert_time)
dlmwrite('generate_data\DFT0_DFT1_25_83_Hilbert_256_148_5168.txt', DFT0_DFT1_Hilbert_time)