figure; 
D = load("data/twomoons/true.mat");
xstar = D.true;
for i = 1:149
    D = load(sprintf("data/twomoons/xpos_%d.mat", i));
    xq = D.xpos;

    clf
    hold on; 
    h2 = scatter(xstar(:, 1), xstar(:, 2), 'ko');
    h2.MarkerFaceAlpha = .02;
    h2.MarkerEdgeAlpha = .02;

    h1 = scatter(xq(:, 1), xq(:, 2), 'rx');
    h1.SizeData = 32;
    h1.MarkerFaceColor = 'r';
    h1.MarkerEdgeColor = 'r';
    h1.MarkerFaceAlpha = .1;
    h1.MarkerEdgeAlpha = .1;

    title("black: true posterior sample, red: transported samples");
        
    grid on
    axis([-1,1,-1,1])
    drawnow

    F(i) = getframe(gcf) ;
end

% create the video writer with 1 fps
writerObj = VideoWriter('twomoons', 'MPEG-4');
writerObj.FrameRate = 10;
  % set the seconds per image
% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(F)
    % convert the image to a frame
    frame = F(i) ;    
    writeVideo(writerObj, frame);
end
% close the writer object
close(writerObj);
