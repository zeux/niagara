
#if __APPLE__

#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

id getLayerFromWindow(id window)
{
	CAMetalLayer* layer = [CAMetalLayer layer];
	NSWindow* nsWindow = window;
	nsWindow.contentView.layer = layer;
	nsWindow.contentView.wantsLayer = YES;

	return layer;
}

#endif
