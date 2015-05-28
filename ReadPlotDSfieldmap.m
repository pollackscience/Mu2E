% read in DS field map and plot field components to study variations with
% position in the magnet, to study field mapping requirements
%inputfile = uigetfile('FieldMapData_1760_v5/Mu2e_DSMap.txt');
%inputfile.print();
%M = importdata(inputfile,' ');
disp('starting prog');
M = importdata('FieldMapData_1760_v5/Mu2e_DSMap.txt');
disp('finished import');

%
%    disp(M.colheaders{1,3})
%    disp(M.data(1:5,1:3))
%
% coordinates are in mm
 dX= 25; 
nX = 97;
 dY= 25; 
nY = 49;
 dZ= 25; 
nZ = 521; 
%
%(except the first Z point is missing)
%
X=M(:,1);
Y=M(:,2);
Z=M(:,3);
Bx=M(:,4);
By=M(:,5);
Bz=M(:,6);
X0=-4000.;
% transverse X coordinate with respect to DS center
Xds=X-X0;
% radial coordinate from DS axis
R=sqrt(Xds.^2+Y.^2);
Bperp=sqrt(Bx.^2+By.^2);
%
[n m]=size(Z);
%Bz0=0*Bz;
%Bz7=Bz0;
%Bperp0=Bz0;
%Bperp7=Bz0;
%
% STUDY Bz and Br versus Y for X=0 plane
Ymax = 700.;
NYpts = Ymax/dY+1;
Z0 = zeros(nZ,NYpts);
Bz0= zeros(nZ,NYpts);
Bperp0= zeros(nZ,NYpts);
Ys = zeros(nZ,NYpts);
Xs = zeros(nZ,NYpts);
gradBzdz = zeros(nZ,NYpts);
gradBzdr = zeros(nZ,NYpts);
gradBrdz = zeros(nZ,NYpts);
gradBrdr = zeros(nZ,NYpts);

% Field Map loops over all Z for each Y
% Select only field values for X=0 (or closest value, which is 4 mm)
iY = 0;
for Ynth=0:dY:Ymax
    ij=0;
    iY = iY+1;
    ifig1 = 2*iY-1;
    ifig2 = 2*iY;
    for j=1:n

         if (Xds(j)<5) && (Xds(j)>-5) && (Y(j)==Ynth)
% 
            ij=ij+1;
            Z0(ij, iY)=Z(j);
            Ys(ij,iY) = Ynth;
            Xs(ij,iY) = Xds(j);
            Bz0(ij,iY)=Bz(j);
            Bperp0(ij,iY)=Bperp(j);
         end
%
    end

    % figure(ifig1)
    % plot(Z0,Bz0,'b*')
    %    title(['DS Bz vs Z at X=0 at Y=',num2str(Ynth)])
    %    xlabel('Z,mm');
    %    ylabel('Bz,T');
            
    % figure(ifig2)
    % plot(Z0,Bperp0,'r+')
    %    title(['DS Br vs Z at X=0 at Y=',num2str(Ynth)])
    %    xlabel('Z,mm');
    %    ylabel('Br,T');
            
end   
%
     figure(1)
        title('DS Bz vs Z  vs Y at X=0')
        xlabel('Z,mm');
        ylabel('Y,mm');
        zlabel('Bz,T')
        hold on
    for kY = 1:NYpts
        plot3(Z0(:,kY),Ys(:,kY),Bz0(:,kY),'b'),view(45,30)
    end
        grid on
        hold off
            
     figure(2)
        title('DS Br vs Z  vs Y at X=0')
        xlabel('Z,mm');
        ylabel('Y,mm');
        zlabel('Br,T')
        hold on
    for kY = 1:NYpts
        plot3(Z0(:,kY),Ys(:,kY),Bperp0(:,kY),'r'),view(45,30)
    end
        grid on
        hold off
     
% RADIAL FIELD IN THE TRACKER REGION
     figure(3)
        title('DS Br vs Y  vs Z(tracker) at X=0')
        xlabel('Z,mm');
        ylabel('Y,mm');
        zlabel('Br,T')
        hold on
        
% Z limits for plots of uniform DS region
     iZmin = 210;
     iZmax = 350;
% Z limits of tracker position
     iTZmin = 215;
     iTZmax = 345;
     
     NZpts = (iZmax-iZmin)+1;
    for kZ = 1:NZpts
        indexZ = iZmin+kZ-1;
        plot3(Z0(indexZ,:),Ys(indexZ,:),Bperp0(indexZ,:),'g'),view(10,30)
    end
    % mark the edges of the tracker position with special markers
    plot3(Z0(215,:),Ys(215,:),Bperp0(215,:),'m*'),view(10,30)
    plot3(Z0(345,:),Ys(345,:),Bperp0(345,:),'kx'),view(10,30)
        grid on
        hold off

% AXIAL FIELD IN THE TRACKER REGION
        
     figure(4)
        title('DS Bz vs Y  vs Z(tracker) at X=0')
        xlabel('Z,mm');
        ylabel('Y,mm');
        zlabel('Bz,T')
        hold on
        

     NZpts = (iZmax-iZmin)+1;
    for kZ = 1:NZpts
        indexZ = iZmin+kZ-1;
        plot3(Z0(indexZ,:),Ys(indexZ,:),Bz0(indexZ,:),'g'),view(10,30)
    end
    % mark the edges of the tracker position with special markers
    plot3(Z0(215,:),Ys(215,:),Bz0(215,:),'m*'),view(10,30)
    plot3(Z0(345,:),Ys(345,:),Bz0(345,:),'kx'),view(10,30)
        grid on
        hold off
%======================================================================%        
% Calculate maximum gradients within tracker region

dBZdrMax = 0.;
dBRdrMax = 0.;
ZY_dBZdrMax = [0 0];
ZY_dBRdrMax = [0 0];
%     NZpts = (iTZmax-iTZmin)+1;
     NZpts = (iZmax-iZmin)+1;
     NYptsMax = Ymax/dY+1;
     
% loop over Y at each Z, calculate and find max radial gradients

    for kZ = iZmin:iZmax   % iTZmin+1:iTZmax
        
            BYlast = Bperp0(kZ,1);
            BZlast = Bz0(kZ,1);
            for kY = 2:NYptsMax
                dBRdr = (Bperp0(kZ,kY)-BYlast)/dY;
                dBZdr = (Bz0(kZ,kY)-BZlast)/dY;
                BYlast = Bperp0(kZ,kY);
                BZlast = Bz0(kZ,kY);
                gradBzdr(kZ,kY) = dBZdr * 10000.; % units G/mm
                gradBrdr(kZ,kY) = dBRdr * 10000.; % units G/mm
                if (abs(dBRdr)>dBRdrMax) && (kZ>iTZmin-1) && (kZ<iTZmax+1)
                    dBRdrMax = abs(dBRdr);
                    ZY_dBRdrMax = [Z0(kZ,kY) Ys(kZ,kY)] ; %coords of max grad
                end
                if (abs(dBZdr)>dBZdrMax) && (kZ>iTZmin-1) && (kZ<iTZmax+1)
                    dBZdrMax = abs(dBZdr);
                    ZY_dBZdrMax = [Z0(kZ,kY) Ys(kZ,kY)];  %coords of max grad
                end
            end

    end
    % convert T to Gauss
    % maximum radial gradients in G/mm within Tracker region:
    dZdRmax = dBZdrMax*10000. % units G/mm
    ZY_dZdr = ZY_dBZdrMax
    dYdRmax = dBRdrMax*10000. % units G/mm
    ZY_dRdr = ZY_dBRdrMax
    
    %- - - - - - - - - - - - - - - - - - - - - - - - - - %
    dBZdzMax = 0.;
    dBRdzMax = 0.;
    ZY_dBZdzMax = [0 0];
    ZY_dBRdzMax = [0 0];

    % loop over Z at each Y, calculate and find max axial gradient
        for kY = 1:NYptsMax
        
            BYlast = Bperp0(iZmin,kY);
            BZlast = Bz0(iZmin,kY);
            for kZ = iZmin+1:iZmax % iTZmin+1:iTZmax
                dBRdz = (Bperp0(kZ,kY)-BYlast)/dZ;
                dBZdz = (Bz0(kZ,kY)-BZlast)/dZ;
                BYlast = Bperp0(kZ,kY);
                BZlast = Bz0(kZ,kY);
                gradBzdz(kZ,kY) = dBZdz * 10000.; % units G/mm
                gradBrdz(kZ,kY) = dBRdz * 10000.; % units G/mm
               if (abs(dBRdz)>dBRdzMax) && (kZ>iTZmin-1) && (kZ<iTZmax+1)
                    dBRdzMax = abs(dBRdz);
                    ZY_dBRdzMax(1,1) = Z0(kZ,kY);
                    ZY_dBRdzMax(1,2) = Ys(kZ,kY);
                end
                if (abs(dBZdz)>dBZdzMax) && (kZ>iTZmin-1) && (kZ<iTZmax+1)
                    dBZdzMax = abs(dBZdz);
                    ZY_dBZdzMax(1,1) = Z0(kZ,kY);
                    ZY_dBZdzMax(1,2) = Ys(kZ,kY);
                end
            end

        end
    % convert T to Gauss
    % maximum radial gradients in G/mm within Tracker region:
    dZdZmax = dBZdzMax*10000. % units G/mm
    ZY_dZdz = ZY_dBZdzMax
    dYdZmax = dBRdzMax*10000. % units G/mm
    ZY_dRdz = ZY_dBRdzMax

%======================================================================%
% plot the gradients in the uniform tracker region
%
% first dBz/dz
       figure(11)
        title('DS dBz/dz vs Y  vs Z(tracker) at X=0')
        xlabel('Z,mm');
        ylabel('Y,mm');
        zlabel('dBz/dz,G/mm')
        hold on

        
    for kZ = iZmin+1:iZmax
        plot3(Z0(kZ,:),Ys(kZ,:),gradBzdz(kZ,:),'b'),view(10,30)
    end
    % mark the edges of the tracker position with special markers
    plot3(Z0(iTZmin,:),Ys(iTZmin,:),gradBzdz(iTZmin,:),'m*'),view(10,30)
    plot3(Z0(iTZmax,:),Ys(iTZmax,:),gradBzdz(iTZmax,:),'rx'),view(10,30)
        grid on
        hold off

%---------------------------------------------------------------------%
% second dBz/dr

               figure(12)
        title('DS dBz/dr vs Y  vs Z(tracker) at X=0')
        xlabel('Z,mm');
        ylabel('Y,mm');
        zlabel('dBz/dr,G/mm')
        hold on
        
    for kZ = iZmin+1:iZmax
        plot3(Z0(kZ,:),Ys(kZ,:),gradBzdr(kZ,:),'b'),view(10,30)
    end
    % mark the edges of the tracker position with special markers
    plot3(Z0(iTZmin,:),Ys(iTZmin,:),gradBzdr(iTZmin,:),'m*'),view(10,30)
    plot3(Z0(iTZmax,:),Ys(iTZmax,:),gradBzdr(iTZmax,:),'rx'),view(10,30)
        grid on
        hold off
        
%---------------------------------------------------------------------%
% third dBr/dz

               figure(13)
        title('DS dBr/dz vs Y  vs Z(tracker) at X=0')
        xlabel('Z,mm');
        ylabel('Y,mm');
        zlabel('dBr/dz,G/mm')
        hold on
        
    for kZ = iZmin+1:iZmax
        plot3(Z0(kZ,:),Ys(kZ,:),gradBrdz(kZ,:),'b'),view(10,30)
    end
    % mark the edges of the tracker position with special markers
    plot3(Z0(iTZmin,:),Ys(iTZmin,:),gradBrdz(iTZmin,:),'m*'),view(10,30)
    plot3(Z0(iTZmax,:),Ys(iTZmax,:),gradBrdz(iTZmax,:),'rx'),view(10,30)
        grid on
        hold off

%---------------------------------------------------------------------%
% fourth dBr/dr

               figure(14)
        title('DS dBr/dr vs Y  vs Z(tracker) at X=0')
        xlabel('Z,mm');
        ylabel('Y,mm');
        zlabel('dBr/dr,G/mm')
        hold on
        
    for kZ = iZmin+1:iZmax
        plot3(Z0(kZ,:),Ys(kZ,:),gradBrdr(kZ,:),'b'),view(10,30)
    end
    % mark the edges of the tracker position with special markers
    plot3(Z0(iTZmin,:),Ys(iTZmin,:),gradBrdr(iTZmin,:),'m*'),view(10,30)
    plot3(Z0(iTZmax,:),Ys(iTZmax,:),gradBrdr(iTZmax,:),'rx'),view(10,30)
        grid on
        hold off
        
