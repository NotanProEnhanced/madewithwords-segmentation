// Loads the Meta Pixel and/or Pinterest tag base code, each only if its id is
// set (see /pixel-config.js — must load before this script). With no ids
// configured, this file does nothing. Fires PageView/page on load either way,
// and exposes window.trackPurchase() for a page that confirms its own sale
// client-side (currently just /success — the gallery's digital download has
// no HTML page to fire from, so that path relies on the server-side
// Conversions API in _social_capi_event() instead; see app/main.py).
(function () {
  "use strict";
  var T = window.TRACKING || {};

  if (T.meta) {
    /* eslint-disable */
    !function(f,b,e,v,n,t,s){if(f.fbq)return;n=f.fbq=function(){n.callMethod?
    n.callMethod.apply(n,arguments):n.queue.push(arguments)};if(!f._fbq)f._fbq=n;
    n.push=n;n.loaded=!0;n.version='2.0';n.queue=[];t=b.createElement(e);t.async=!0;
    t.src=v;s=b.getElementsByTagName(e)[0];s.parentNode.insertBefore(t,s)}(window,
    document,'script','https://connect.facebook.net/en_US/fbevents.js');
    /* eslint-enable */
    try {
      window.fbq("init", T.meta);
      window.fbq("track", "PageView");
    } catch (e) { /* noop */ }
  }

  if (T.pinterest) {
    /* eslint-disable */
    !function(e){if(!window.pintrk){window.pintrk=function(){window.pintrk.queue.push(
    Array.prototype.slice.call(arguments))};var n=window.pintrk;n.queue=[],n.version="3.0";
    var t=document.createElement("script");t.async=!0,t.src=e;var r=document.getElementsByTagName(
    "script")[0];r.parentNode.insertBefore(t,r)}}("https://s.pinimg.com/ct/core.js");
    /* eslint-enable */
    try {
      window.pintrk("load", T.pinterest);
      window.pintrk("page");
    } catch (e) { /* noop */ }
  }

  // valueUsd: dollars (not cents). eventId: pass the same id the server-side CAPI
  // call used for this sale (e.g. the Stripe session id) so Meta/Pinterest dedupe
  // a sale that fires on both sides instead of double-counting it.
  window.trackPurchase = function (valueUsd, currency, contentId, eventId) {
    currency = currency || "USD";
    try {
      if (T.meta && window.fbq) {
        window.fbq("track", "Purchase", {
          value: valueUsd, currency: currency, content_ids: [contentId], content_type: "product"
        }, eventId ? { eventID: eventId } : undefined);
      }
    } catch (e) { /* noop */ }
    try {
      if (T.pinterest && window.pintrk) {
        window.pintrk("track", "checkout", {
          value: valueUsd, currency: currency, line_items: [{ product_id: contentId }],
          event_id: eventId
        });
      }
    } catch (e) { /* noop */ }
  };
})();
