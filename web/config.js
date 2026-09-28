// Supabase project URL and PUBLISHABLE (public) key - safe to ship in
// client-side JS by design, same as any other Supabase frontend. This key
// can ONLY execute the specific api_* functions granted to it in
// database/migrations/009_lock_down_anon_access.sql - it has zero direct
// table access (no SELECT/INSERT/UPDATE/DELETE on any table), so exposing
// it here carries none of the risk a service-role key would.
const SUPABASE_URL = "https://egemorkltmcxqadugotl.supabase.co";
const SUPABASE_PUBLISHABLE_KEY = "sb_publishable_KFb58URRj-4k2c2RtjVY7g_CgE4Y1_e";
