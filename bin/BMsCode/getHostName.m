function hostName = getHostName()
[~,hostName]= system('hostname'); 
hostName = hostName(1:end-1); % system(hostname) returns an empty char in the end    