function [ff]=ff_pp_194_46(x,chang)


fe=x;
len_sig = length(fe);
a = ceil((len_sig-chang)/(chang-1));%����֡��
% ��sig����pad
b = a*(chang-1)+chang;%����a֡�ƣ�chang֡�������ܵ���
fe = [fe;zeros(b-len_sig,1)];%��ԭ������������0���

all_duan=[1,chang];%�洢ÿ֡�������յ�λ��
for i=1:(chang-1)
    st = all_duan(i,2)-(chang-a);%��ȡ��ǰ֡�����λ��
    en = st+chang;%��ȡ��ǰ֡���յ�λ��
    d = [st,en];
    all_duan=[all_duan;d];
end

first_fe=fe(all_duan(1,1):all_duan(1,2));%��ȡ��һ֡����������ֵΪchang��
fea=[first_fe];
for j=2:chang
    ind_s=all_duan(j,1);%��ȡ��һ֡�����
    ind_e=all_duan(j,2);%��ȡ��һ֡���յ�
    zj=fe(ind_s:ind_e-1);%��ȡ��ǰ֡����������ֵ
    fea=[fea;zj];%�洢��ǰ֡����������ֵ
end
ff=fea';
end

