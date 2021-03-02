clear
% For simple example
% Lav disse til funktion af billedstørrelse og filterstørrelse. 
n = 4; 
m = 4;
% Create n-by-n circulant matrix
B_block = gallery('circul',1:n)';
% Create nm-by-nm block circulant matrix
B_all = repmat(B_block,m);


q = (1:n^2)';


ts = zeros(n);
for i = 1:n
    ts(i,:) = fft(B_block(:,1));
end

ws = zeros(n);
for i = 0:(n-1)
    ws(i+1,:) = fft(q((i*n)+1:(i+1)*n));
end

out = zeros(n*n, 1);
for j = 1:n 
    x = 0;
    s = circshift(flip((1:n)),j);
    for i = 1:n
            x = x + (ts(s(i),:).*ws(i,:));
    end
    out((j*n)-n +1:(j*n)) = ifft(x)';
   
end



mat = B_all*q;


