classdef plt_funcs
    methods(Static)

        % ===== visualize B matrix - magnitude & phase =====
        function [mag, phase] = B_mag_phase(b)
            n = size(b,1);
            k = size(b,2)/2;
            
            mag = zeros(n,k);
            phase = zeros(n,n,k);
            
            % magnitude
            for i=1:n       %node
                for j=1:k   %oscillator
                    mag(i,j) = norm([b(i,j*2-1), b(i,j*2)]);
                end
            end
            
            % phases
            for g=1:k
                for i=1:n
                    for j=1:n
                        v1 = [b(i,g*2-1), b(i,g*2)];
                        v2 = [b(j,g*2-1), b(j,g*2)];
                        if (mag(i,g)>0.15 && mag(j,g)>0.15)
                            if atan2(v1(1),v1(2))-atan2(v2(1),v2(2))>0
                                phase(i,j,g) = mod(atan2(v1(1),v1(2))-atan2(v2(1),v2(2)),pi);
                            else
                                phase(i,j,g) = mod((atan2(v2(1),v2(2))-atan2(v1(1),v1(2))),-pi);
                            end
                        end
                    end
                end
            end
        end

        % ===== layout for 64 electrodes [propofol data] =====
        function out = layout(ename,value)
            nm = ['','','','',"FP1",'FPz','FP2','','','','',...
                '','','','AF7','AF3','AFz','AF4','AF8','','','',...
                '','F7','F5','F3','F1','Fz','F2','F4','F6','F8','',...
                '','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10',...
                '','T7','C5','C3','C1','Cz','C2','C4','C6','T8','',...
                'TP9','TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','TP10',...
                '','P7','P5','P3','P1','Pz','P2','P4','P6','P8','',...
                '','','','PO7','PO3','POz','PO4','PO8','','','',...
                '','','','','O1','Oz','O2','','','',''];

            out = zeros(9,11);
            for i=1:9
                for j=1:11
                    for k=1:length(ename)
                        if nm((i-1)*11+j) == ename(k)
                            out(i,j) = value(k);
                            break
                        end
                    end
                end
            end
        end
    end
end

    