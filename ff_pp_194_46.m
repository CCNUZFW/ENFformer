function [ff]=ff_pp_194_46(x,chang)


fe=x;
len_sig = length(fe);
a = ceil((len_sig-chang)/(chang-1));%计算帧移
% 对sig进行pad
b = a*(chang-1)+chang;%计算a帧移，chang帧数所需总点数
fe = [fe;zeros(b-len_sig,1)];%对原特征点数进行0填充

all_duan=[1,chang];%存储每帧的起点和终点位置
for i=1:(chang-1)
    st = all_duan(i,2)-(chang-a);%获取当前帧的起点位置
    en = st+chang;%获取当前帧的终点位置
    d = [st,en];
    all_duan=[all_duan;d];
end

first_fe=fe(all_duan(1,1):all_duan(1,2));%获取第一帧的所有特征值为chang个
fea=[first_fe];
for j=2:chang
    ind_s=all_duan(j,1);%获取下一帧的起点
    ind_e=all_duan(j,2);%获取下一帧的终点
    zj=fe(ind_s:ind_e-1);%获取当前帧的所有特征值
    fea=[fea;zj];%存储当前帧的所有特征值
end
ff=fea';
end

