function fsrj = fsr_eeg(y, embed_dim)

T = length(y);
y_embed = zeros(embed_dim,T-embed_dim+1);

for d = 1:embed_dim
    y_embed(d,:) = y(d:end-embed_dim+d);
end

s = svd(y_embed);
s = sort(s,'descend');
fsrj = cumsum(s)/sum(s);

end