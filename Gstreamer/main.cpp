/**
 * RTSP Server menggunakan GStreamer
 *
 * Program ini membuat RTSP server yang dapat melayani stream video/audio.
 * Mendukung beberapa mount point (stream):
 *   - /test   : Video test pattern dengan audio tone
 *   - /camera : Video dari webcam (jika tersedia)
 *   - /file   : Stream dari file video (jika file tersedia)
 *
 * Cara menjalankan:
 *   ./rtsp_server [port] [file_video]
 *
 * Cara membuka stream (gunakan VLC atau ffplay):
 *   rtsp://localhost:8554/test
 *   rtsp://localhost:8554/camera
 *   rtsp://localhost:8554/file
 */

#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <glib.h>
#include <iostream>
#include <string>
#include <csignal>
#include <cstring>

// ─── Global loop untuk signal handling ───────────────────────────────────────
static GMainLoop *g_main_loop = nullptr;

// ─── Signal handler untuk graceful shutdown ──────────────────────────────────
static void signal_handler(int signum) {
    std::cout << "\n[INFO] Menerima sinyal " << signum
              << " - Menghentikan server..." << std::endl;
    if (g_main_loop)
        g_main_loop_quit(g_main_loop);
}

// ─── Callback: Client terhubung ──────────────────────────────────────────────
static void on_client_connected(GstRTSPServer * /*server*/,
                                GstRTSPClient *client,
                                gpointer       /*user_data*/) {
    GstRTSPConnection *conn = gst_rtsp_client_get_connection(client);
    if (conn) {
        const gchar *ip = gst_rtsp_connection_get_ip(conn);
        std::cout << "[CLIENT] Terhubung dari: " << (ip ? ip : "unknown")
                  << std::endl;
    }
}

// ─── Callback: Media siap diputar ────────────────────────────────────────────
static void on_media_prepared(GstRTSPMediaFactory * /*factory*/,
                               GstRTSPMedia        * /*media*/,
                               gpointer              user_data) {
    const gchar *mount = (const gchar *)user_data;
    std::cout << "[MEDIA] Stream siap: " << mount << std::endl;
}

// ─── Buat factory untuk test pattern (videotestsrc + audiotestsrc) ────────────
static GstRTSPMediaFactory *create_test_factory() {
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();

    // Pipeline: video test pattern + audio test tone dikemas dalam RTP
    const gchar *pipeline =
        "( "
        "videotestsrc pattern=ball is-live=true ! "
        "video/x-raw,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! "
        "x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
        "rtph264pay name=pay0 pt=96 "
        "  "
        "audiotestsrc wave=sine freq=440 is-live=true ! "
        "audio/x-raw,rate=44100,channels=1 ! "
        "audioconvert ! "
        "avenc_aac ! "
        "rtpmp4apay name=pay1 pt=97 "
        ")";

    gst_rtsp_media_factory_set_launch(factory, pipeline);
    gst_rtsp_media_factory_set_shared(factory, TRUE);  // Semua client berbagi stream
    gst_rtsp_media_factory_set_latency(factory, 200);  // 200ms buffer latency

    return factory;
}

// ─── Buat factory untuk webcam ────────────────────────────────────────────────
static GstRTSPMediaFactory *create_camera_factory() {
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();

    // Coba v4l2src dulu, fallback ke videotestsrc jika tidak ada kamera
    const gchar *pipeline =
        "( "
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! "
        "x264enc tune=zerolatency bitrate=800 speed-preset=superfast ! "
        "rtph264pay name=pay0 pt=96 "
        ")";

    gst_rtsp_media_factory_set_launch(factory, pipeline);
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    gst_rtsp_media_factory_set_latency(factory, 100);

    return factory;
}

// ─── Buat factory untuk file video ───────────────────────────────────────────
static GstRTSPMediaFactory *create_file_factory(const std::string &filepath) {
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();

    // Stream file video dengan looping
    std::string pipeline =
        "( "
        "filesrc location=\"" + filepath + "\" ! "
        "decodebin name=dec "
        "dec. ! queue ! videoconvert ! "
        "x264enc tune=zerolatency bitrate=1000 speed-preset=superfast ! "
        "rtph264pay name=pay0 pt=96 "
        "dec. ! queue ! audioconvert ! audioresample ! "
        "avenc_aac ! "
        "rtpmp4apay name=pay1 pt=97 "
        ")";

    gst_rtsp_media_factory_set_launch(factory, pipeline.c_str());
    gst_rtsp_media_factory_set_shared(factory, FALSE); // Tiap client dapat stream sendiri
    gst_rtsp_media_factory_set_latency(factory, 300);

    return factory;
}

// ─── Buat factory untuk low-latency test (hanya video, tanpa audio) ──────────
static GstRTSPMediaFactory *create_lowlatency_factory() {
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();

    const gchar *pipeline =
        "( "
        "videotestsrc pattern=smpte is-live=true ! "
        "video/x-raw,width=1280,height=720,framerate=25/1 ! "
        "videoconvert ! "
        "x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast "
        "   key-int-max=30 ! "
        "rtph264pay name=pay0 pt=96 config-interval=-1 "
        ")";

    gst_rtsp_media_factory_set_launch(factory, pipeline);
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    gst_rtsp_media_factory_set_latency(factory, 0); // Ultra low latency

    return factory;
}

// ─── Print info server ────────────────────────────────────────────────────────
static void print_server_info(const std::string &port,
                               const std::string &filepath) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║           GStreamer RTSP Server v1.0                 ║\n";
    std::cout << "╠══════════════════════════════════════════════════════╣\n";
    std::cout << "║  Port    : " << port
              << std::string(42 - port.size(), ' ') << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════╣\n";
    std::cout << "║  Stream URLs yang tersedia:                          ║\n";
    std::cout << "║                                                      ║\n";
    std::cout << "║  [TEST]    rtsp://localhost:" << port << "/test"
              << std::string(22 - port.size(), ' ') << "║\n";
    std::cout << "║            Video test pattern + audio 440Hz          ║\n";
    std::cout << "║                                                      ║\n";
    std::cout << "║  [CAMERA]  rtsp://localhost:" << port << "/camera"
              << std::string(20 - port.size(), ' ') << "║\n";
    std::cout << "║            Webcam /dev/video0                        ║\n";
    std::cout << "║                                                      ║\n";
    std::cout << "║  [HD]      rtsp://localhost:" << port << "/hd"
              << std::string(23 - port.size(), ' ') << "║\n";
    std::cout << "║            720p low-latency test stream              ║\n";

    if (!filepath.empty()) {
        std::cout << "║                                                      ║\n";
        std::cout << "║  [FILE]    rtsp://localhost:" << port << "/file"
                  << std::string(22 - port.size(), ' ') << "║\n";
        std::cout << "║            " << filepath.substr(0, 40)
                  << std::string(40 - std::min((int)filepath.size(), 40), ' ')
                  << "  ║\n";
    }

    std::cout << "╠══════════════════════════════════════════════════════╣\n";
    std::cout << "║  Cara membuka stream:                                ║\n";
    std::cout << "║    ffplay rtsp://localhost:" << port << "/test"
              << std::string(22 - port.size(), ' ') << "║\n";
    std::cout << "║    vlc    rtsp://localhost:" << port << "/test"
              << std::string(22 - port.size(), ' ') << "║\n";
    std::cout << "╠══════════════════════════════════════════════════════╣\n";
    std::cout << "║  Tekan Ctrl+C untuk menghentikan server              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    // Parse argumen
    std::string port     = "8554";
    std::string filepath = "";

    if (argc >= 2) port     = argv[1];
    if (argc >= 3) filepath = argv[2];

    // Inisialisasi GStreamer
    gst_init(&argc, &argv);

    // Setup signal handler
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ── Buat RTSP Server ──
    GstRTSPServer *server = gst_rtsp_server_new();
    gst_rtsp_server_set_service(server, port.c_str());

    // ── Daftarkan signal client-connected ──
    g_signal_connect(server, "client-connected",
                     G_CALLBACK(on_client_connected), nullptr);

    // ── Ambil mount points ──
    GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(server);

    // ── Mount /test : Video test pattern + audio ──
    {
        GstRTSPMediaFactory *factory = create_test_factory();
        g_signal_connect(factory, "media-prepared",
                         G_CALLBACK(on_media_prepared),
                         (gpointer)"/test");
        gst_rtsp_mount_points_add_factory(mounts, "/test", factory);
        std::cout << "[INFO] Mount point /test didaftarkan\n";
    }

    // ── Mount /camera : Webcam ──
    {
        GstRTSPMediaFactory *factory = create_camera_factory();
        g_signal_connect(factory, "media-prepared",
                         G_CALLBACK(on_media_prepared),
                         (gpointer)"/camera");
        gst_rtsp_mount_points_add_factory(mounts, "/camera", factory);
        std::cout << "[INFO] Mount point /camera didaftarkan\n";
    }

    // ── Mount /hd : Low-latency HD stream ──
    {
        GstRTSPMediaFactory *factory = create_lowlatency_factory();
        g_signal_connect(factory, "media-prepared",
                         G_CALLBACK(on_media_prepared),
                         (gpointer)"/hd");
        gst_rtsp_mount_points_add_factory(mounts, "/hd", factory);
        std::cout << "[INFO] Mount point /hd didaftarkan\n";
    }

    // ── Mount /file : File video (opsional) ──
    if (!filepath.empty()) {
        GstRTSPMediaFactory *factory = create_file_factory(filepath);
        g_signal_connect(factory, "media-prepared",
                         G_CALLBACK(on_media_prepared),
                         (gpointer)"/file");
        gst_rtsp_mount_points_add_factory(mounts, "/file", factory);
        std::cout << "[INFO] Mount point /file didaftarkan: " << filepath << "\n";
    }

    g_object_unref(mounts);

    // ── Attach server ke default main context ──
    guint server_id = gst_rtsp_server_attach(server, nullptr);
    if (server_id == 0) {
        std::cerr << "[ERROR] Gagal menjalankan RTSP server pada port "
                  << port << std::endl;
        g_object_unref(server);
        return EXIT_FAILURE;
    }

    // ── Print info ──
    print_server_info(port, filepath);

    // ── Jalankan main loop ──
    g_main_loop = g_main_loop_new(nullptr, FALSE);
    g_main_loop_run(g_main_loop);

    // ── Cleanup ──
    std::cout << "[INFO] Membersihkan resource..." << std::endl;
    g_source_remove(server_id);
    g_object_unref(server);
    g_main_loop_unref(g_main_loop);
    gst_deinit();

    std::cout << "[INFO] Server dihentikan. Sampai jumpa!\n";
    return EXIT_SUCCESS;
}